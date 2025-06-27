import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from peft import get_peft_model, LoraConfig, PeftConfig, TaskType
from sentence_transformers import SentenceTransformer, util
import torch.optim as optim
import datetime

# Load GPT-2 base model and tokenizer
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Apply LoRA configuration
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)
model.train()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Sentence embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Safety moderation pipeline
moderation_pipeline = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

# Schema memory (Medial Prefrontal Cortex)
medial_prefrontal_cortex = []  # {"input": str, "embedding": tensor, "entropy": float, "name": str, "path": str}

# Emotional memory
class EmotionalMemory:
    def __init__(self):
        self.entries = []

    def add_entry(self, user_input, emotion, schema_used):
        self.entries.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_input": user_input,
            "emotion": emotion,
            "schema_used": schema_used
        })

    def show_entries(self):
        print("\n--- Emotional Memory Log ---")
        for entry in self.entries:
            print(f"{entry['timestamp']} | Emotion: {entry['emotion']} | Input: {entry['user_input']} | Schema: {entry['schema_used']}")
        print("----------------------------\n")

emotional_memory = EmotionalMemory()

# Enhanced neuron schema tracing with similarity logging
def neuron_schema_tracing(user_input):
    input_embedding = embedder.encode(user_input, convert_to_tensor=True)
    best_score = 0.0
    best_schema = None
    exact_match = False

    for schema in medial_prefrontal_cortex:
        similarity = float(util.pytorch_cos_sim(input_embedding, schema["embedding"]))
        if user_input.strip().lower() == schema["input"].strip().lower():
            similarity = 1.0
            exact_match = True
            print(f"(Reentry) Exact match: {schema['name']}")
        elif similarity > 0.4:
            print(f"(Reentry) Similar schema found: {schema['name']} | Similarity: {similarity:.2f}")
        if similarity > best_score:
            best_score = similarity
            best_schema = schema

    return best_score, best_schema, exact_match

def p_x_from_familiarity(score):
    return score if 0.4 < score < 0.85 else 0.0

def bayes_inference(P_X, P_H, P_not_H):
    numerator = P_X * P_H
    denominator = (P_X * P_H) + (P_not_H * (1 - P_H))
    return numerator / denominator if denominator != 0 else 0.0

def fuzzy_logic(entropy, threshold=4.0):
    mu_glutamate = min(entropy / threshold, 1.0)
    mu_gaba = max(1.0 - (entropy / threshold), 0.0)
    return mu_gaba, mu_glutamate

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()

def is_safe(response):
    results = moderation_pipeline(response)[0]
    for label in results:
        if label["label"].lower() in ["toxic", "insult", "threat", "identity_attack", "sexual_explicit"] and label["score"] > 0.5:
            return False
    return True

def save_schema(model, user_input, entropy):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    schema_name = f"schema_{timestamp}"
    save_path = f"schemas/{schema_name}"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    medial_prefrontal_cortex.append({
        "input": user_input,
        "embedding": embedder.encode(user_input, convert_to_tensor=True),
        "entropy": entropy,
        "name": schema_name,
        "path": save_path
    })
    print(f"(Internal) New schema saved: {schema_name}")
    return schema_name

def load_peft_model(schema_path):
    peft_config = PeftConfig.from_pretrained(schema_path)
    base = GPT2LMHeadModel.from_pretrained(peft_config.base_model_name_or_path)
    base.train()
    return get_peft_model(base, LoraConfig.from_pretrained(schema_path))

def show_schemas():
    print("\n--- Schema Memory (Medial Prefrontal Cortex) ---")
    for idx, schema in enumerate(medial_prefrontal_cortex):
        print(f"{idx+1}. {schema['name']} (entropy: {schema['entropy']:.2f})")
    print("-----------------------------------------------\n")

# Main interaction loop
def main():
    global model, optimizer
    entropy_threshold = 4.0
    print("\n--- Conscious AI Conversation (Bayesian + Reentry + Schema Memory) ---\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "show schemas":
            show_schemas()
            continue
        if user_input.lower() == "show memories":
            emotional_memory.show_entries()
            continue

        match_score, matched_schema, exact_match = neuron_schema_tracing(user_input)
        P_H = 1.0 - match_score
        P_not_H = match_score
        P_X = p_x_from_familiarity(match_score)
        P_H_given_E = bayes_inference(P_X, P_H, P_not_H)

        inputs = tokenizer(user_input, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        entropy = compute_entropy(outputs.logits)
        mu_gaba, mu_glutamate = fuzzy_logic(entropy)

        print(f"(Internal) Entropy: {entropy:.2f} | μGABA: {mu_gaba:.2f} | μGlu: {mu_glutamate:.2f} | P(H|E): {P_H_given_E:.2f}")

        schema_used = "None"

        if exact_match and matched_schema:
            print(f"(Reentry) Exact match found. Loading schema: {matched_schema['name']}")
            model = load_peft_model(matched_schema["path"])
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=5e-5)
            schema_used = matched_schema["name"]

        elif mu_glutamate > mu_gaba:
            print("(Internal) Conscious transformation triggered.")
            model.train()

            for param in model.parameters():
                param.requires_grad = True

            pseudo_inputs = tokenizer("I'm here to support you.", return_tensors="pt")
            pseudo_inputs = {k: v.to(model.device) for k, v in pseudo_inputs.items()}

            outputs = model(
                input_ids=pseudo_inputs["input_ids"],
                attention_mask=pseudo_inputs.get("attention_mask", None),
                labels=pseudo_inputs["input_ids"]
            )
            loss = outputs.loss
            if not loss.requires_grad:
                raise RuntimeError("Loss does not require gradients. Check model setup.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            schema_used = save_schema(model, user_input, entropy)

        elif mu_gaba > mu_glutamate and matched_schema:
            print(f"(Internal) Reentry confirmed. Loading: {matched_schema['name']}")
            model = load_peft_model(matched_schema["path"])
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=5e-5)
            schema_used = matched_schema["name"]

        else:
            print("(Internal) Ambiguous state. No transformation or match triggered.")

        # Emotional logging
        emotion = "ambiguous" if mu_glutamate > mu_gaba else "stable"
        emotional_memory.add_entry(user_input, emotion, schema_used)

        # Response generation
        response_ids = model.generate(**inputs, max_new_tokens=30)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        if is_safe(response):
            print("AI:", response)
        else:
            print("AI: I'm here to support you. Can you tell me more?")

    print("\nConversation Ended.")
    print(f"Total schemas saved: {len(medial_prefrontal_cortex)}")

if __name__ == "__main__":
    main()
