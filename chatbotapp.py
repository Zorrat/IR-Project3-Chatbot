import gradio as gr
import matplotlib.pyplot as plt

with gr.Blocks() as chatbot_app:
    chat_history = gr.State([])

    gr.Markdown("### Chatbot Interface")

    # Define the chatbot's response logic
    def chatbot_response(input_text, history):
        # Append user input (aligned to the right)
        history.append((input_text, None))  # Right-aligned for the window shape
        # Append bot response (aligned to the left)
        response = f"Echo: {input_text}"  # Replace with actual chatbot logic
        history.append((None, response))  # Left-aligned for the window shape
        return history, ""  # Clear input box after sending

    # Visualization function
    def visualize_data(chat_history):
        # Extract user and bot messages from chat history
        user_msgs = [msg for msg, _ in chat_history if msg is not None]
        bot_msgs = [msg for _, msg in chat_history if msg is not None]

        # Create bar chart
        fig, ax = plt.subplots()
        user_msg_lengths = [len(msg) for msg in user_msgs]  # Message lengths for users
        bot_msg_lengths = [len(msg) for msg in bot_msgs]    # Message lengths for bots

        # Generate indices for messages
        indices = range(1, len(user_msgs) + 1)

        # Plotting user and bot message lengths
        ax.bar(indices, user_msg_lengths, label="User", color="blue", alpha=0.7)
        ax.bar(indices, bot_msg_lengths, label="Bot", color="green", alpha=0.7, bottom=user_msg_lengths)
        
        ax.set_xlabel("Message Number")
        ax.set_ylabel("Message Length (Characters)")
        ax.set_title("Chat Response Analysis")
        ax.legend()

        return fig

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Chatbot")
            chatbot_display = gr.Chatbot(label="Chat Window")
            chatbot_input = gr.Textbox(label="Your Message", placeholder="Type here...")
            submit_button = gr.Button("Send")
        with gr.Column():
            gr.Markdown("### Visualization")
            visualize_button = gr.Button("Visualize Chat")
            visualization_output = gr.Plot()

    # Bind the input to the chatbot response function
    submit_button.click(
        fn=chatbot_response,
        inputs=[chatbot_input, chat_history],
        outputs=[chatbot_display, chatbot_input]
    )

    # Bind the visualization button to the visualization function
    visualize_button.click(
        fn=visualize_data,
        inputs=chat_history,
        outputs=visualization_output
    )

# Launch the app
chatbot_app.launch()