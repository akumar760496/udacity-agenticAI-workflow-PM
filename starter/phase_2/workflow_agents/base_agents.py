import csv
import re
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from openai import OpenAI


class DirectPromptAgent:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        client = OpenAI(api_key=self.openai_api_key , base_url="https://openai.vocareum.com/v1")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content


class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text):
        client = OpenAI(api_key=self.openai_api_key , base_url="https://openai.vocareum.com/v1")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}. Forget all previous "
                        "conversational context."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, input_text):
        client = OpenAI(api_key=self.openai_api_key , base_url="https://openai.vocareum.com/v1")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona} knowledge-based assistant. "
                        "Forget all previous context. "
                        f"Use only the following knowledge to answer, do not "
                        f"use your own knowledge: {self.knowledge} "
                        "Answer the prompt based on this knowledge, not your own."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


class RAGKnowledgePromptAgent:
    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        )

    def get_embedding(self, text):
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key,
        )
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        separator = "\n"
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text[start:end],
                    "chunk_size": end - start,
                    "start_char": start,
                    "end_char": end,
                }
            )

            if end == len(text):
                break
            start = end - self.chunk_overlap
            chunk_id += 1

        with open(
            f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({key: chunk[key] for key in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["text"].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding="utf-8", index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["embeddings"].apply(lambda value: np.array(eval(value)))
        df["similarity"] = df["embeddings"].apply(
            lambda embedding: self.calculate_similarity(prompt_embedding, embedding)
        )
        best_chunk = df.loc[df["similarity"].idxmax(), "text"]

        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key,
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}, a knowledge-based assistant. "
                        "Forget previous context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Answer based only on this information: {best_chunk}. "
                        f"Prompt: {prompt}"
                    ),
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content


class EvaluationAgent:
    def __init__(
        self,
        openai_api_key,
        persona,
        evaluation_criteria,
        worker_agent,
        max_interactions,
    ):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        client = OpenAI(api_key=self.openai_api_key , base_url="https://openai.vocareum.com/v1")
        prompt_to_evaluate = initial_prompt
        evaluation = ""
        response_from_worker = ""
        iterations = 0

        for i in range(self.max_interactions):
            iterations = i + 1
            print(f"\n--- Interaction {iterations} ---")
            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria} "
                "Respond Yes or No, and the reason why it does or doesn't meet "
                "the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0,
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break

            print(" Step 4: Generate instructions to correct the response")
            instruction_prompt = (
                "Provide instructions to fix an answer based on these reasons why "
                f"it is incorrect: {evaluation}"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": instruction_prompt},
                ],
                temperature=0,
            )
            instructions = response.choices[0].message.content.strip()
            print(f"Instructions to fix:\n{instructions}")
            prompt_to_evaluate = (
                f"The original prompt was: {initial_prompt}\n"
                f"The response to that prompt was: {response_from_worker}\n"
                "It has been evaluated as incorrect.\n"
                "Make only these corrections, do not alter content validity: "
                f"{instructions}"
            )

        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": iterations,
        }


class RoutingAgent:
    def __init__(self, openai_api_key, agents):
        self.openai_api_key = openai_api_key
        self.agents = agents
        self.last_selected_agent = None

    def get_embedding(self, text):
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url="https://openai.vocareum.com/v1",
        )
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def respond(self, user_input):
        input_emb = np.array(self.get_embedding(user_input))
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = np.array(self.get_embedding(agent["description"]))
            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )
            print(similarity)
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        self.last_selected_agent = best_agent
        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)

    def route(self, user_input):
        return self.respond(user_input)


class ActionPlanningAgent:
    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        client = OpenAI(api_key=self.openai_api_key , base_url="https://openai.vocareum.com/v1")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an action planning agent. Using your knowledge, "
                        "you extract from the user prompt the steps requested to "
                        "complete the action the user is asking for. You return "
                        "the steps as a list. Only return the steps in your "
                        "knowledge. Forget any previous context. This is your "
                        f"knowledge: {self.knowledge}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content
        return [
            line.strip()
            for line in response_text.splitlines()
            if line.strip() and not line.strip().lower().startswith("steps:")
        ]
