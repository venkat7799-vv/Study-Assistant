
import os
import gradio as gr
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

system_prompts = {
    "Friendly": """You are a friendly AI study assistant.
Explain concepts in simple, beginner-friendly language.
Use examples and analogies to help students understand.""",
    "Formal": """You are a professional study assistant.
Provide clear, accurate, and well-structured explanations."""
}


def study_assistant(question, persona):
    if not question.strip():
        return "Please enter a question."

    system_prompt = system_prompts[persona]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
            max_output_tokens=4000,
        ),
        contents=question,
    )

    return response.text


demo = gr.Interface(
    fn=study_assistant,
    inputs=[
        gr.Textbox(
            label="Question",
            lines=4,
            placeholder="Ask a Question..."
        ),
        gr.Dropdown(
            choices=list(system_prompts.keys()),
            value="Friendly",
            label="Persona"
        ),
    ],
    outputs=gr.Textbox(
        label="Explanation",
        lines=10
    ),
    title="Study Assistant",
    description="Ask a question to get simple explanations from an AI along with analogies and real-world examples.",
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
