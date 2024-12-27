import gradio as gr

# from beartype import beartype
from emopred.prediction import predict_emotion


with gr.Blocks(gr.themes.Glass()) as app:
    gr.Markdown(
        """
    # FACIAL EXPRESSION PREDICTION

    ### QUICK GUIDE
    This application processes video sequences to analyze facial expressions and classify them into three main categories: Positive, Neutral, and Negative. Here are some key functionalities and tips to help you get the most out of this tool:
    1. Upload or capture a video using the **Video Source** section.
    2. The processed video output will be displayed in the **Processed Video** section, showing detected faces and their predicted expressions.

    ### HOW IT WORKS
    - The system uses face detection and deep learning models to recognize and classify facial expressions frame-by-frame.
    - Detected expressions are overlaid on the video, and real-time feedback is provided during processing.

    **NOTE**: The application is built to automatically process the video upon upload and display the results, including highlighted faces and corresponding emotion labels.
    """
    )
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Video Source")
        with gr.Column():
            output_video = gr.Video(
                label="Processed Video", streaming=True, autoplay=True)

    video.change(fn=predict_emotion, inputs=[video], outputs=[output_video])

if __name__ == "__main__":
    app.launch()
