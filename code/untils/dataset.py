#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import re
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
import transformers
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoTokenizer,
    AutoModel,
)
from modelscope import Model
from swift.tuners import Swift


# -------------------------
# last-token pooling
# -------------------------
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


# ---------------------------------------------------------
# augment_ent — generate name/desc/sum 
# ---------------------------------------------------------
def augment_ent(data_dir, output_dir, model_dir):
    import json, transformers, torch
    from tqdm import tqdm

    # Load entity file
    with open(data_dir, "r", encoding="utf-8") as f:
        entities = json.load(f)

    data_ids = [e.get("id") for e in entities]
    data_qid = [e.get("qid") for e in entities]
    data_name = [e.get("entity_name", "") for e in entities]
    data_des = [e.get("desc", "") for e in entities]

    # Load LLaMA model
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # System prompt
    system = "you are a helpful assistant!"

    # YOUR PROMPT (unchanged)
    PROMPT = (
        "Please generate a one-sentence summary for the given entity, including entity name and description.\n\n"
        "Entity name: {entity_name}\n"
        "Entity description: {entity_des}\n\n"
        "Try your best to summarize the main content of the given entity. And generate a short summary in 1 sentence for it.\n"
        "Summary:"
    )

    # Load previous progress (if any)
    try:
        with open(output_dir, "r", encoding="utf-8") as f:
            out_list = json.load(f)
    except:
        out_list = []

    processed_ids = set(o.get("ids") for o in out_list if o.get("sum", "") != "")

    for i in tqdm(range(len(data_ids))):
        ent_id = data_ids[i]

        # Regenerate if sum is empty OR missing
        if ent_id in processed_ids:
            continue

        ent_name = data_name[i]
        ent_desc = data_des[i]

        record = {"ids": ent_id, "qid": data_qid[i], "name": ent_name, "des": ent_desc}

        user_text = PROMPT.format(entity_name=ent_name, entity_des=ent_desc)

        # Build Llama-3 chat-formatted prompt
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

        try:
            prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            prompt = user_text

        try:
            outputs = pipeline(
                prompt,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=pipeline.model.config.eos_token_id
            )

            raw = outputs[0]["generated_text"]

            # Remove prompt from beginning
            if raw.startswith(prompt):
                gen = raw[len(prompt):].strip()
            else:
                gen = raw.strip()

            # Extract after "Summary:"
            if "Summary:" in gen:
                gen = gen.split("Summary:")[-1].strip()

            gen = " ".join(gen.split())

            # -------------------------------------------------
            # CLEANUP: Remove unwanted prefix from LLaMA output
            # -------------------------------------------------
            unwanted_prefixes = [
                f"Here is a one-sentence summary for the entity \"{ent_name}\":",
                f"Here is a one-sentence summary for the entity '{ent_name}':",
                f"Here is a one-sentence summary for the entity {ent_name}:",
                f"Here is a one-sentence summary for {ent_name}:",
                f"Here is a one-sentence summary for the entity \"{ent_name}\"",
                f"Here is a one-sentence summary for {ent_name}"
            ]

            for p in unwanted_prefixes:
                if gen.startswith(p):
                    gen = gen[len(p):].strip()

            # Global fallback (if model changes phrasing)
            if "Here is a one-sentence summary" in gen:
                parts = gen.split(":", 1)
                if len(parts) > 1:
                    gen = parts[1].strip()

            # -------------------------------------------------

            # Extract a clean 1-sentence summary
            dot = gen.find(".")
            if 0 <= dot < 400:
                one_sent = gen[: dot + 1]
            else:
                one_sent = gen[:300]

            # Guarantee entity name appears
            if ent_name not in one_sent:
                one_sent = f"{ent_name} — {one_sent}"

            record["sum"] = one_sent

        except Exception as e:
            print(f"error {ent_id}: {e}")
            record["sum"] = ""

        out_list.append(record)

        # Save every 100 steps
        if i % 100 == 0:
            with open(output_dir, "w", encoding="utf-8") as f:
                json.dump(out_list, f, ensure_ascii=False, indent=2)

    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)


# -------------------------
# run_emb — QID-based entity embeddings
# -------------------------
def run_emb(model_dir, data_dir, embed_dir, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.to("cuda").eval()

    with open(data_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeds = []
    for ent in tqdm(data, desc="Embedding entities"):
        qid = ent.get("qid")
        if qid is None:
            continue  # skip entities without QID

        text = (ent.get("name", "") + ":" + ent.get("sum", "")).replace("\n", " ")

        batch = tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**batch)
            emb = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])[0]
            emb = F.normalize(emb, dim=0)  # IMPORTANT

        embeds.append({
            "qid": qid,
            "emb": emb.tolist()
        })

    with open(embed_dir, "w", encoding="utf-8") as f:
        json.dump(embeds, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(embeds)} entity embeddings to {embed_dir}")


# -------------------------
# augment_men_img (robust, minimal-change replacement)
# -------------------------
def augment_men_img(mentions_dir, save_dir, model_id, image_dir):
    """
    Robust image-augmentation that:
      - keeps your PROMPT format (as requested),
      - loads LLaVA model/processor (works with offline model files),
      - tries multiple ways to resolve the image filename from the JSON entry:
          1) direct join(image_dir, imgPath)
          2) if image_dir looks like a prefix (ends with 'train_' etc) use that prefix
          3) extract digits from imgPath (e.g. 'mention_54.jpg' -> 54) and search
             the standard folders train_image/ valid_image/ test_image for files
             named train_54.jpg / valid_54.jpg / test_54.jpg (or test_54.jpg etc).
      - writes back "des_llava" field (empty string if generation failed)
      - logs missing files and a short summary at the end.
    """
    import json
    import os
    import re
    import torch
    from tqdm import tqdm
    from PIL import Image
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    # ---------- Load mentions ----------
    with open(mentions_dir, "r", encoding="utf-8") as f:
        mentions = json.load(f)

    # ---------- Load model & processor ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, low_cpu_mem_usage=True
        ).to(device)
    except Exception as e:
        print(f"ERROR: failed to load LLaVA model/processor from {model_id}: {e}")
        raise

    # ---------- Your requested PROMPT (kept exactly, minor whitespace normalized) ----------
    PROMPT = """The target entity is a "{mention_category}" named "{mention_name}".
The image describes "{mention_context}".
Introduce the "{mention_category}" named "{mention_name}". Answer follow the format: "The {mention_name} refer to..."
Only generate an introduction to the target entity, not a description of the image.
"""

    # ---------- Helpers to resolve image path ----------
    def extract_digits(filename):
        m = re.search(r"(\d+)", filename)
        return m.group(1) if m else None

    def try_join_path(base, fname):
        # return absolute path if exists, else None
        p = os.path.join(base, fname)
        if os.path.exists(p):
            return p
        return None

    def find_image_file(img_filename):
        """
        Attempts multiple strategies to find image file on disk.
        Returns absolute path or None.
        Strategies (in order):
          A) if img_filename is absolute and exists -> return it
          B) join(image_dir, img_filename)
          C) if image_dir ends with 'train_' or 'valid_' or 'test_' -> use it as prefix
             e.g. image_dir=/.../train_image/train_  -> prefix + <num>.jpg
          D) search in sibling folders under mention_images (train_image/ valid_image/ test_image)
             using numeric ID from img_filename: train_<id>.jpg, valid_<id>.jpg, test_<id>.jpg
          E) fallback: scan image_dir recursively for a file containing the digits or the basename
        """
        # A: absolute
        if os.path.isabs(img_filename) and os.path.exists(img_filename):
            return img_filename

        # B: join with provided image_dir (common case if image_dir is folder)
        if os.path.isdir(image_dir):
            p = try_join_path(image_dir, img_filename)
            if p:
                return p

        # C: image_dir might be a prefix like ".../train_image/train_"
        if image_dir.endswith("_") or image_dir.endswith("train") or image_dir.endswith("valid") or image_dir.endswith("test"):
            # try various ways
            base_dir = image_dir
            # if ends with 'train_' or 'train', adapt
            if image_dir.endswith("_"):
                # prefix mode
                digits = extract_digits(img_filename)
                if digits:
                    candidate = image_dir + digits + os.path.splitext(img_filename)[1]
                    if os.path.exists(candidate):
                        return candidate
            else:
                # could be folder path, try join
                p = try_join_path(image_dir, img_filename)
                if p:
                    return p

        # D: use numeric id and search common subfolders next to image_dir
        digits = extract_digits(img_filename)
        if digits:
            # try possible parent mention_images directory guesses
            # if image_dir contains 'mention_images', use it as root; else try siblings
            if "mention_images" in image_dir:
                root = image_dir[: image_dir.find("mention_images") + len("mention_images")]
            else:
                # try image_dir parent/grandparent
                root = os.path.abspath(os.path.join(image_dir, "..", ".."))
            candidates = []
            # common folders inside mention_images
            common_subs = ["train_image", "valid_image", "test_image"]
            for sub in common_subs:
                folder = os.path.join(root, sub)
                if os.path.isdir(folder):
                    # possible names
                    for prefix in ("train_", "valid_", "test_"):
                        fname = f"{prefix}{digits}.jpg"
                        p = os.path.join(folder, fname)
                        if os.path.exists(p):
                            return p
                    # also try prefix matching the sub folder
                    # e.g. if sub == 'test_image' try test_<digits>.jpg
                    prefix2 = sub.split("_")[0] + "_"  # train_, valid_, test_
                    fname2 = f"{prefix2}{digits}.jpg"
                    p2 = os.path.join(folder, fname2)
                    if os.path.exists(p2):
                        return p2

        # E: fallback recursive scan limited to image_dir (if it's a directory)
        if os.path.isdir(image_dir):
            # look for exact basename match or containing digits
            base = os.path.basename(img_filename)
            digits = extract_digits(img_filename)
            for root, _, files in os.walk(image_dir):
                for f in files:
                    if f == base:
                        return os.path.join(root, f)
                    if digits and digits in f:
                        return os.path.join(root, f)

        # not found
        return None

    # ---------- Loop through mentions and generate ----------
    missing_count = 0
    generated_count = 0
    for i in tqdm(range(len(mentions))):
        ent = mentions[i]
        # use sentence or context field (both supported)
        context = ent.get("context", ent.get("sentence", ""))
        # Use the name/category fields in JSON; fallback to mentions/entities as needed
        mention_name = ent.get("name") or ent.get("mentions") or ent.get("entities") or ""
        mention_category = ent.get("category") or "entity"

        # ensure there is an imgPath entry
        img_file = ent.get("imgPath", "")
        if not img_file:
            mentions[i]["des_llava"] = ""
            missing_count += 1
            continue

        # build prompt & chat wrapper
        prompt_text = PROMPT.format(
            mention_category=mention_category,
            mention_name=mention_name,
            mention_context=context,
        )
        # Create the chat-style wrapper expected by LLaVA if necessary
        # Many LLaVA processors accept just [INST] <image>\n... [/INST]
        chat_prompt = f"[INST] <image>\n{prompt_text} [/INST]"

        # find image path robustly
        img_path = find_image_file(img_file)
        if img_path is None:
            # not found: log and set empty
            print(f"Missing image for index {i} id={ent.get('id')} imgPath='{img_file}' --> tried multiple lookups")
            mentions[i]["des_llava"] = ""
            missing_count += 1
            continue

        # load image and prepare inputs
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(chat_prompt, image, return_tensors="pt").to(device)
        except Exception as e:
            print(f"Image load/processor error at index {i} path={img_path}: {e}")
            mentions[i]["des_llava"] = ""
            missing_count += 1
            continue

        # generate
        try:
            output = model.generate(**inputs, max_new_tokens=100)
            resp = processor.decode(output[0], skip_special_tokens=True)
            # clean response
            resp_text = " ".join(resp.split())
            mentions[i]["des_llava"] = resp_text
            generated_count += 1
        except Exception as e:
            print(f"Generation error at index {i} path={img_path}: {e}")
            mentions[i]["des_llava"] = ""
            missing_count += 1
            continue

    # ---------- Save updated JSON ----------
    with open(save_dir, "w", encoding="utf-8") as f:
        json.dump(mentions, f, ensure_ascii=False, indent=2)

    print(f"augment_men_img finished: generated={generated_count}, missing_or_failed={missing_count}, total={len(mentions)}")

# -------------------------
# augment_men_text (updated)
# -------------------------
def augment_men_text(data_dir, output_dir, model_dir):

    # Load mention file
    with open(data_dir, "r", encoding="utf-8") as f:
        entities = json.load(f)

    # Load text-generation model
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    system = "you are a helpful assistant!"

    PROMPT = """Please make a brief description in 1 sentence for the entity under the background of context.

### Entity
The entity is a {category}.
Name: {mention_name}
Context: {mention_context}

# Description (Describe the entity without referring to context.)
"""

    # Load previous outputs if available
    try:
        with open(output_dir, "r", encoding="utf-8") as f:
            out_list = json.load(f)
        processed_ids = {item["id"] for item in out_list}
    except:
        out_list = []
        processed_ids = set()

    # Loop over all mentions
    for ent in tqdm(entities):

        # Skip if already processed
        if ent["id"] in processed_ids:
            continue

        # If image-based description exists, skip generating text
        if "des_llava" in ent:
            # Still store a placeholder so pipeline remains consistent
            out_list.append(ent)
            continue

        # Extract context
        context = ent.get("context", ent.get("sentence", ""))

        # Build text prompt
        text = PROMPT.format(
            category=ent.get("category", ""),
            mention_name=ent.get("name", ""),
            mention_context=context,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]

        try:
            prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            prompt = text  # fallback

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Run model
        try:
            outputs = pipeline(
                prompt,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=pipeline.model.config.eos_token_id,
            )
            generated = outputs[0]["generated_text"][len(prompt):].strip()

        except Exception as e:
            print(f"Error in ID {ent['id']}: {e}")
            generated = ""

        # 🔥 REQUIRED FIX: Save text description into "des"
        ent["des"] = generated.replace("\n", " ")

        # Add record to output list
        out_list.append(ent)

    # Save output
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)

    print(f"✔ Text augmentation completed. Saved to {output_dir}")


# -------------------------
# runtopK — QID-based candidate reranking (paper-aligned)
# -------------------------
def runtopK(K, model_dir, database_emb, database_sum,
            mention_dir, mention_topK_dir, max_length):

    print("⚠️ Using embedding-based Top-K reranking (QID-based)")

    import json
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.to("cuda").eval()

    # -------- load entity embeddings (QID → emb) --------
    with open(database_emb, "r", encoding="utf-8") as f:
        ents_emb = json.load(f)

    emb_map = {
        e["qid"]: torch.tensor(e["emb"])
        for e in ents_emb
    }

    # -------- load mentions --------
    with open(mention_dir, "r", encoding="utf-8") as f:
        mentions = json.load(f)

    correct = 0

    for i in tqdm(range(len(mentions)), desc="Top-K reranking"):
        name = mentions[i]["name"]
        context = mentions[i].get("context", mentions[i].get("sentence", ""))
        text = context + "\n" + name

        batch = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**batch)
            mention_emb = last_token_pool(
                outputs.last_hidden_state,
                batch["attention_mask"]
            )[0]
            mention_emb = F.normalize(mention_emb, dim=0).cpu()

        scores = []
        valid_cands = []

        for cand in mentions[i]["cands"]:
            if cand not in emb_map:
                continue

            ent_emb = F.normalize(emb_map[cand], dim=0)
            score = torch.dot(mention_emb, ent_emb)

            scores.append(score)
            valid_cands.append(cand)

        if scores:
            scores = torch.stack(scores)
            topk_idx = torch.topk(scores, min(K, len(scores))).indices.tolist()
            mentions[i]["new_cands"] = [valid_cands[j] for j in topk_idx]
        else:
            mentions[i]["new_cands"] = []

        # QID-based accuracy
        if mentions[i].get("qid") in mentions[i]["new_cands"]:
            correct += 1

    with open(mention_topK_dir, "w", encoding="utf-8") as f:
        json.dump(mentions, f, ensure_ascii=False, indent=2)

    print(f"Top-{K} accuracy: {correct / len(mentions):.4f}")

# -------------------------
# infer — QID-based, ckpt-safe, CLEAN OUTPUT (FINAL)
# -------------------------
def infer(model_id, ckpt_id, max_length, database_sum, mention_topK_dir, res_output_dir):

    import json
    import re
    import torch
    from tqdm import tqdm
    import transformers
    from transformers import AutoTokenizer
    from modelscope import Model
    from swift import Swift

    device = "cuda"

    # -------------------------
    # Load base LLM
    # -------------------------
    model = Model.from_pretrained(
        model_id,
        device_map="auto",
        max_length=max_length
    )

    # -------------------------
    # OPTIONAL: load Swift checkpoint
    # -------------------------
    if ckpt_id:
        model = Swift.from_pretrained(
            model,
            ckpt_id,
            inference_mode=True,
            max_length=max_length
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    # -------------------------
    # Load entity summaries (QID-based)
    # -------------------------
    with open(database_sum, "r", encoding="utf-8") as f:
        ents = json.load(f)

    ent_map = {
        e["qid"]: {
            "name": e.get("name", ""),
            "sum": e.get("sum", "")
        }
        for e in ents
        if "qid" in e
    }

    # -------------------------
    # Load Top-K mentions
    # -------------------------
    with open(mention_topK_dir, "r", encoding="utf-8") as f:
        mentions = json.load(f)

    PROMPT = """
You are an expert in knowledge graph entity linking.
Select the best matching entity for the given mention.

### Mention
Name: {mention_name}
Context: {mention_context}
Category: {mention_category}
Description: {mention_des}

### Entity table
0. {entity_0}
1. {entity_1}
2. {entity_2}
3. {entity_3}
4. {entity_4}

Only output the serial number (0-4).
The most matched serial number is:
"""

    prompts = []
    truth = []

    # -------------------------
    # Build prompts
    # -------------------------
    for m in tqdm(mentions, desc="Preparing prompts"):
        cands = m.get("new_cands", [])

        # ground truth index
        try:
            t = cands.index(m["qid"])
        except:
            t = -1
        truth.append(t)

        entity_table = []
        for qid in cands[:5]:
            if qid in ent_map:
                entity_table.append(
                    ent_map[qid]["name"] + ": " + ent_map[qid]["sum"]
                )
            else:
                entity_table.append("")

        context = m.get("context", m.get("sentence", ""))
        desc = m.get("des_llava", m.get("des", ""))

        text = PROMPT.format(
            mention_name=m.get("name", ""),
            mention_context=context,
            mention_category=m.get("category", ""),
            mention_des=desc,
            entity_0=entity_table[0] if len(entity_table) > 0 else "",
            entity_1=entity_table[1] if len(entity_table) > 1 else "",
            entity_2=entity_table[2] if len(entity_table) > 2 else "",
            entity_3=entity_table[3] if len(entity_table) > 3 else "",
            entity_4=entity_table[4] if len(entity_table) > 4 else "",
        )

        prompts.append(text)

    # -------------------------
    # Run LLM inference (CLEAN OUTPUT)
    # -------------------------
    preds = []
    for text in tqdm(prompts, desc="LLM inference"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]

        try:
            prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            prompt = text

        try:
            out = pipeline(prompt, max_new_tokens=64)
            raw = out[0]["generated_text"]

            # remove prompt echo
            if raw.startswith(prompt):
                raw = raw[len(prompt):]

            # extract ONLY a digit 0–4
            match = re.findall(r"\b[0-4]\b", raw)
            pred = match[0] if match else ""
        except:
            pred = ""

        preds.append(pred)

    # -------------------------
    # Save clean results
    # -------------------------
    results = []
    for i in range(len(preds)):
        results.append({
            "pred": preds[i],   # clean: "0"–"4"
            "true": truth[i]
        })

    with open(res_output_dir, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # -------------------------
    # Accuracy (NOT meaningful yet)
    # -------------------------
    correct = 0
    for r in results:
        try:
            p = int(r["pred"])
        except:
            p = -1
        if p == r["true"] and r["true"] != -1:
            correct += 1

    print("Final Accuracy:", correct / len(results))
