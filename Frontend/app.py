import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class ChatbotModel:
    def __init__(self, model_path="../Model/emergency-gpt2-final"):
        """Initialize the DistilGPT-2 model and tokenizer"""
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
            print(f"Model type: {self.model.config.model_type}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(self, user_input, max_length=200, temperature=0.8, top_p=0.92):
        """Generate response from user input"""
        if not user_input.strip():
            return "Please enter a message."
        
        try:
            prompt = f"<|user|> {user_input.strip()}<|assistant|>"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=False
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=50,
                    do_sample=True,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "<|assistant|>" in generated_text:
                response = generated_text.split("<|assistant|>")[-1].strip()
            else:
                response = generated_text.replace(prompt, "").strip()
            
            if not response or len(response) < 10:
                response = "I understand your question. Could you provide more details so I can give you better emergency guidance?"
            
            response = response.replace("<|endoftext|>", "").strip()
            
            return response
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error. Please try rephrasing your question."


print("Initializing Emergency Companion Chatbot...")
chatbot_model = ChatbotModel()

def chat_function(message, history, max_length, temperature, top_p):
    """Process chat messages"""
    response = chatbot_model.generate_response(
        message,
        max_length=int(max_length),
        temperature=temperature,
        top_p=top_p
    )
    return response


custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.emergency-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
"""

with gr.Blocks(title="Emergency Companion Chatbot", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML(
        """
        <div class="emergency-header">
            <h1>Emergency Companion Chatbot</h1>
            <p style="font-size: 16px; margin-top: 10px;">Powered by Fine-tuned DistilGPT-2 Model</p>
            <p style="font-size: 14px; opacity: 0.9;">Get immediate first aid guidance and emergency support</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=True,
                avatar_images=(None, "🤖"),
                type="messages",
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Emergency Question",
                    placeholder="e.g., How do I treat a cut? What to do for a fever?",
                    lines=2,
                    scale=4
                )
                submit = gr.Button("Send 📤", variant="primary", scale=1)
            
            with gr.Row():
                clear = gr.Button("Clear Chat 🗑️", variant="secondary")
                
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Settings")
            
            max_length = gr.Slider(
                minimum=50,
                maximum=300,
                value=150,
                step=10,
                label="Max Response Length",
                info="Maximum new tokens to generate"
            )
            
            temperature = gr.Slider(
                minimum=0.3,
                maximum=1.5,
                value=0.8,
                step=0.1,
                label="Temperature",
                info="Higher = more creative, Lower = more focused"
            )
            
            top_p = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=0.92,
                step=0.05,
                label="Top P (Nucleus Sampling)",
                info="Diversity of token selection"
            )
            
            gr.Markdown(
                """
                ---
                ### Usage Tips
                - **For medical emergencies**: Be specific about symptoms
                - **Temperature**: Use 0.7-0.9 for emergency advice
                - **Lower temperature**: More factual, focused responses
                - **Higher temperature**: More varied phrasing
                
                ---
                ### Disclaimer
                This is an AI assistant for **informational purposes only**.
                For serious emergencies, always call emergency services (911).
                """
            )
            
            gr.Markdown(
                """
                ---
                ### Model Info
                - **Architecture**: DistilGPT-2
                - **Parameters**: 82M
                - **BLEU Score**: 0.18
                - **ROUGE-L**: 0.30
                """
            )
    
    def respond(message, chat_history, max_len, temp, top_p_val):
        """Handle user message and generate bot response"""
        if not message.strip():
            return chat_history, ""
        
        bot_response = chat_function(message, chat_history, max_len, temp, top_p_val)
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_response})
        
        return chat_history, ""
    
    def clear_conversation():
        """Clear the chat history"""
        return [], ""
    
    submit.click(
        respond,
        inputs=[msg, chatbot, max_length, temperature, top_p],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, max_length, temperature, top_p],
        outputs=[chatbot, msg]
    )
    
    clear.click(
        clear_conversation,
        outputs=[chatbot, msg]
    )
    
    gr.Examples(
        examples=[
            "How do I treat a severe cut?",
            "What should I do if someone is choking?",
            "Snake bite emergency - what are the steps?",
            "I have a high fever, what to do?",
            "How to treat a sprained ankle?",
            "Severe burn treatment steps",
            "Someone is drowning, help!",
            "How to perform CPR?",
            "Nosebleed won't stop",
            "Heat stroke symptoms and treatment",
        ],
        inputs=msg,
        label="📋 Example Emergency Questions"
    )
    
    gr.Markdown(
        """
        ---
        <center>
        <p style="color: #666; font-size: 12px;">
        Emergency Companion Chatbot v1.0 | Trained on 4,095 Emergency Response Samples
        </p>
        </center>
        """
    )

if __name__ == "__main__":
    print("Launching Emergency Companion Chatbot Interface")
    print("-"*70)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )