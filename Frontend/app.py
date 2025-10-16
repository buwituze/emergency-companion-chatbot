import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class ChatbotModel:
    def __init__(self, model_path="../Model/emergency-companion-model"):
        """Initialize the T5 model and tokenizer"""
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(self, user_input, max_length=512, temperature=0.7, top_p=0.9):
        """Generate response from user input"""
        if not user_input.strip():
            return "Please enter a message."
        
        try:
            # Format input based on how the model was trained
            # Try different formats - adjust based on your training format
            input_text = user_input  # Simple format - just the question
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_beams=2,  # Reduced beams
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # Prevent repetition
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response if it echoes the input
            if response.lower().startswith("answer:"):
                response = response[7:].strip()
            if response.lower() == user_input.lower():
                response = "I understand your question, but I need more context to provide a helpful answer."
            
            return response
        
        except Exception as e:
            return f"Error generating response: {str(e)}"


# Initialize the chatbot model
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


# Create Gradio interface
with gr.Blocks(title="Emergency Companion Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 Emergency Companion Chatbot
        ### Powered by Fine-tuned T5 Model
        
        Ask me anything - I'm here to help with emergency support and companionship.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=True,
                avatar_images=(None, "🤖"),
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2,
                    scale=4
                )
                submit = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear = gr.Button("Clear Conversation")
                
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Settings")
            
            max_length = gr.Slider(
                minimum=50,
                maximum=512,
                value=256,
                step=10,
                label="Max Response Length",
                info="Maximum number of tokens in response"
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
                info="Higher = more creative, Lower = more focused"
            )
            
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top P (Nucleus Sampling)",
                info="Diversity of token selection"
            )
            
            gr.Markdown(
                """
                ---
                ### 💡 Tips
                - Adjust temperature for creativity
                - Lower temperature for factual responses
                - Higher temperature for varied outputs
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
            "I'm feeling anxious, can you help?",
            "What should I do in an emergency?",
            "I need someone to talk to",
            "Can you give me some advice?",
        ],
        inputs=msg,
        label="Example Messages"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )