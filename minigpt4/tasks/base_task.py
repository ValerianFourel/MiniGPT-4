"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
import wandb
#================================
# Stitch Val
import gc
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
#================================
class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        self.cfg = ""

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        self.cfg = cfg
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        outputs = model(samples)
        loss = outputs["loss"]

        return loss, outputs

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        print('accum_grad_iters: ', accum_grad_iters)
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def unscale_gradients(self, scaler, parameters):
        """
        Manually unscale gradients for parameters.

        Args:
            scaler: The GradScaler instance
            parameters: Iterable of parameters with gradients to unscale

        Returns:
            List of unscaled gradients
        """
        if scaler is None:
            # If no scaler is used, just return the gradients as they are
            return [p.grad for p in parameters if p.grad is not None]

        # Get the scale factor from the scaler
        inv_scale = 1. / scaler.get_scale()

        # Unscale each gradient
        unscaled_grads = [p.grad * inv_scale if p.grad is not None else None 
                        for p in parameters]

        return unscaled_grads


    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            #####################
            # We need to empty the cache()
            torch.cuda.empty_cache()
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            # Clear gradients at the beginning of each iteration
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, outputs = self.train_step(model=model, samples=samples)

            # FIX: Make sure we're properly tracking the backward pass with the scaler
            if use_amp:
                # Scale the loss and call backward - this is where inf checks are recorded
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # In _train_inner_loop after computing loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf loss detected at epoch {epoch}, iter {i}. Skipping.")
                continue



            # we now delete the outputs
            del outputs
            use_unscaled_grads = False

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp and use_unscaled_grads:
                    # If you need unscaled gradients for any reason, get them here
                    # before the optimizer step
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    unscaled_grads = self.unscale_gradients(scaler, trainable_params)

                    # Now you can use unscaled_grads for any custom operations
                    # For example, gradient clipping based on unscaled values:
                    # grad_norm = torch.norm(torch.stack([torch.norm(g) for g in unscaled_grads if g is not None]))
                    # if grad_norm > max_norm:
                    #     # Apply clipping logic here

                    # Continue with normal optimizer step
                    try:
                        # Unscale gradients for optimizer step
                        scaler.unscale_(optimizer)

                        # Optional: Clip gradients
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        # Step and update
                        scaler.step(optimizer)
                        scaler.update()
                    except RuntimeError as e:
                        # Handle "Attempting to unscale FP16 gradients" error
                        if "Attempting to unscale FP16 gradients" in str(e):
                            logging.warning("FP16 gradient unscaling error detected. Using manual unscaling.")
                            # Apply manual updates using unscaled_grads if needed
                            with torch.no_grad():
                                for param, grad in zip(trainable_params, unscaled_grads):
                                    if grad is not None:
                                        param.data.add_(grad, alpha=-optimizer.param_groups[0]['lr'])
                        else:
                            # Re-raise other errors
                            raise e
                else:    
                    optimizer.step()

                optimizer.zero_grad()

                # Log to wandb if enabled
                if self.cfg.run_cfg.wandb_log:
                    wandb.log({"epoch": inner_epoch, "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        #####################
        # We need to empty the cache()
        del loss
        del samples
        torch.cuda.empty_cache()
        gc.collect()

        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }



    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
