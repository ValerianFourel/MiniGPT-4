import argparse
import os
import random
import json
import csv
from collections import Counter
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sentence_transformers import SentenceTransformer, util

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub, Conversation, SeparatorStyle

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# ========================================
#             Launch Arguments
# ========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Emotion Recognition Evaluation")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--image-dir", required=True, help="directory containing images to evaluate")
    parser.add_argument("--ground-truth", required=True, help="path to ground truth emotion labels JSON file")
    parser.add_argument("--output", default="emotion_eval_results.csv", help="path to save evaluation results")
    parser.add_argument("--num-images", type=int, default=10, help="number of images to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

# ========================================
#             Model Initialization
# ========================================

def initialize_minigpt4(args):
    print('Initializing MiniGPT-4')
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    CONV_VISION = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                'pretrain_llama2': CONV_VISION_LLama2,
                'pretrain': CONV_VISION}

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    print('MiniGPT-4 Initialization Finished')

    return chat, CONV_VISION

# ========================================
#             Emotion Questions
# ========================================

def generate_emotion_questions(n=50):
    """Generate n diverse questions about emotions in images."""
    base_questions = [
        "What emotion is expressed in this image?",
        "What feeling does this person convey?",
        "How would you describe the emotional state shown here?",
        "What is the primary emotion in this picture?",
        "What mood is captured in this image?",
        "What sentiment does this image express?",
        "How would you characterize the emotional tone here?",
        "What emotional response does this image convey?",
        "What is the affective state shown in this image?",
        "What emotional expression do you see?",
        "What feeling is being communicated here?",
        "How would you label the emotion in this image?",
        "What is the emotional atmosphere in this picture?",
        "What emotional state is depicted?",
        "What feeling is most prominent in this image?",
        "What emotion best describes what's shown here?",
        "What is the emotional content of this image?",
        "How would you categorize the emotion displayed?",
        "What feeling is being portrayed in this image?",
        "What emotional quality is present here?",
        "What is the emotional essence of this image?",
        "What feeling stands out in this picture?",
        "How would you identify the emotion shown?",
        "What emotional signal is being sent?",
        "What is the predominant feeling in this image?",
        "What emotion is being demonstrated here?",
        "How would you classify the emotion in this picture?",
        "What emotional message does this image convey?",
        "What feeling is being exhibited here?",
        "What is the emotional character of this image?",
        "What emotion is most evident in this picture?",
        "How would you name the feeling shown here?",
        "What emotional impression does this image give?",
        "What is the emotional nature of this picture?",
        "What feeling is represented in this image?",
        "How would you describe the emotional vibe here?",
        "What is the emotional tenor of this image?",
        "What feeling is manifested in this picture?",
        "What emotion does this image project?",
        "How would you define the feeling shown here?",
        "What is the emotional undertone in this image?",
        "What feeling is being communicated visually?",
        "What emotion is captured in this picture?",
        "How would you articulate the feeling displayed?",
        "What is the emotional quality of this image?",
        "What feeling is depicted in this scene?",
        "What emotion is illustrated here?",
        "How would you express the feeling shown?",
        "What is the emotional significance of this image?",
        "What feeling is embodied in this picture?"
    ]

    # If we need more than we have, we'll add variations
    if n <= len(base_questions):
        return random.sample(base_questions, n)

    # Otherwise, create variations by adding prefixes
    prefixes = [
        "In your assessment, ",
        "Based on what you see, ",
        "From your perspective, ",
        "Looking at this image, ",
        "In your opinion, ",
        "From your analysis, ",
        "As you observe this, ",
        "In your view, ",
        "According to your interpretation, ",
        "From what you can tell, "
    ]

    questions = base_questions.copy()
    while len(questions) < n:
        prefix = random.choice(prefixes)
        base = random.choice(base_questions)
        new_question = prefix + base[0].lower() + base[1:]
        if new_question not in questions:
            questions.append(new_question)

    return questions[:n]

# ========================================
#             Evaluation Class
# ========================================

class EmotionRecognitionEvaluator:
    def __init__(self, chat, conv_template, sentence_model_name="all-MiniLM-L6-v2"):
        self.chat = chat
        self.conv_template = conv_template

        # Initialize the sentence transformer model for comparing responses
        print(f"Loading sentence transformer model: {sentence_model_name}")
        self.sentence_model = SentenceTransformer(sentence_model_name)

        # Common emotion categories for comparison
        self.emotion_categories = [
            "happy", "sad", "angry", "surprised", "fearful", "disgusted", 
            "neutral", "contempt", "joy", "fear", "disgust", "anger", 
            "sadness", "surprise", "confusion", "excitement", "calm"
        ]

        # Generate diverse questions about emotions
        self.questions = generate_emotion_questions(50)

    def process_image(self, image_path, ground_truth):
        """Process a single image with multiple emotion questions."""
        results = []

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return []

        # Initialize conversation state
        chat_state = self.conv_template.copy()
        img_list = []

        # Upload image
        self.chat.upload_img(img, chat_state, img_list)
        self.chat.encode_img(img_list)

        # Ask each question and collect responses
        for question in self.questions:
            self.chat.ask(question, chat_state)
            response = self.chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=0.2,  # Lower temperature for more consistent responses
                max_new_tokens=50,
                max_length=100
            )[0]

            # Reset conversation for next question to avoid context influence
            chat_state = self.conv_template.copy()
            self.chat.upload_img(img, chat_state, img_list)

            # Store result
            results.append({
                "image": os.path.basename(image_path),
                "question": question,
                "response": response,
                "ground_truth": ground_truth
            })

        return results

    def analyze_responses(self, results):
        """Analyze the responses to determine the most consistent emotion."""
        if not results:
            return None

        # Extract all responses for this image
        responses = [r["response"] for r in results]
        ground_truth = results[0]["ground_truth"]

        # Encode all responses
        response_embeddings = self.sentence_model.encode(responses)

        # Encode emotion categories
        emotion_embeddings = self.sentence_model.encode(self.emotion_categories)

        # Find the closest emotion category for each response
        predicted_emotions = []
        for resp_emb in response_embeddings:
            similarities = util.cos_sim(resp_emb, emotion_embeddings)[0]
            best_match_idx = torch.argmax(similarities).item()
            predicted_emotions.append(self.emotion_categories[best_match_idx])

        # Count occurrences of each predicted emotion
        emotion_counts = Counter(predicted_emotions)
        most_common_emotion = emotion_counts.most_common(1)[0][0]

        # Calculate consistency (percentage of responses with the most common emotion)
        consistency = emotion_counts[most_common_emotion] / len(responses) * 100

        # Check if ground truth is in the predicted emotions
        ground_truth_found = ground_truth.lower() in [e.lower() for e in predicted_emotions]

        # Calculate ground truth accuracy (percentage of responses matching ground truth)
        ground_truth_matches = sum(1 for e in predicted_emotions if e.lower() == ground_truth.lower())
        ground_truth_accuracy = ground_truth_matches / len(responses) * 100

        return {
            "image": results[0]["image"],
            "ground_truth": ground_truth,
            "predicted_emotion": most_common_emotion,
            "consistency": consistency,
            "ground_truth_found": ground_truth_found,
            "ground_truth_accuracy": ground_truth_accuracy,
            "emotion_distribution": dict(emotion_counts)
        }

# ========================================
#             Main Function
# ========================================

def main():
    args = parse_args()
    setup_seeds(args.seed)

    # Initialize MiniGPT-4
    chat, conv_template = initialize_minigpt4(args)

    # Initialize evaluator
    evaluator = EmotionRecognitionEvaluator(chat, conv_template)

    # Load ground truth data
    with open(args.ground_truth, 'r') as f:
        ground_truth_data = json.load(f)

    # Get list of image files
    image_files = [f for f in os.listdir(args.image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                  and f in ground_truth_data]

    if not image_files:
        print(f"No matching images found in {args.image_dir}")
        return

    # Limit number of images if specified
    if args.num_images and args.num_images < len(image_files):
        image_files = random.sample(image_files, args.num_images)

    print(f"Evaluating emotion recognition on {len(image_files)} images...")

    # Process each image
    all_results = []
    analysis_results = []

    for img_file in tqdm(image_files):
        img_path = os.path.join(args.image_dir, img_file)
        ground_truth = ground_truth_data.get(img_file, "unknown")

        # Process image with multiple questions
        results = evaluator.process_image(img_path, ground_truth)
        all_results.extend(results)

        # Analyze consistency of responses
        analysis = evaluator.analyze_responses(results)
        if analysis:
            analysis_results.append(analysis)

            # Print summary for this image
            print(f"\nImage: {img_file}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Predicted Emotion: {analysis['predicted_emotion']}")
            print(f"Consistency: {analysis['consistency']:.1f}%")
            print(f"Ground Truth Accuracy: {analysis['ground_truth_accuracy']:.1f}%")
            print("Emotion Distribution:", end=" ")
            for emotion, count in sorted(analysis['emotion_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"{emotion}: {count}", end=", ")
            print()

    # Save detailed results
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['image', 'ground_truth', 'predicted_emotion', 'consistency', 
                     'ground_truth_accuracy', 'ground_truth_found', 'emotion_distribution']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in analysis_results:
            # Convert emotion_distribution to string for CSV
            result['emotion_distribution'] = str(result['emotion_distribution'])
            writer.writerow(result)

    # Calculate overall metrics
    correct_predictions = sum(1 for r in analysis_results if r['ground_truth'].lower() == r['predicted_emotion'].lower())
    accuracy = correct_predictions / len(analysis_results) * 100 if analysis_results else 0
    avg_consistency = sum(r['consistency'] for r in analysis_results) / len(analysis_results) if analysis_results else 0
    avg_gt_accuracy = sum(r['ground_truth_accuracy'] for r in analysis_results) / len(analysis_results) if analysis_results else 0

    print("\n===== OVERALL RESULTS =====")
    print(f"Total Images: {len(analysis_results)}")
    print(f"Overall Accuracy: {accuracy:.1f}%")
    print(f"Average Consistency: {avg_consistency:.1f}%")
    print(f"Average Ground Truth Accuracy: {avg_gt_accuracy:.1f}%")
    print(f"Detailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
