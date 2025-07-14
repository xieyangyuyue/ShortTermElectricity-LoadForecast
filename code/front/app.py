import gradio as gr

# 定义一个处理函数，模拟对输入数据的处理
def process_data(text, radio, checkbox, slider, file, image, video, audio, dataframe):
    # 这里只是简单地返回输入数据，实际应用中可以进行复杂的处理
    return (
        text,
        radio,
        checkbox,
        slider,
        file.name if file else "No file uploaded",
        image,
        video,
        audio,
        dataframe
    )


# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# Gradio 组件示例")

    with gr.Row():
        text_input = gr.Textbox(label="文本输入")
        radio_input = gr.Radio(["选项1", "选项2", "选项3"], label="单选按钮")
        checkbox_input = gr.CheckboxGroup(["选项A", "选项B", "选项C"], label="复选框")
        slider_input = gr.Slider(minimum=0, maximum=100, label="滑块")

    with gr.Row():
        file_input = gr.File(label="文件上传")
        image_input = gr.Image(label="图像上传")
        video_input = gr.Video(label="视频上传")
        audio_input = gr.Audio(label="音频上传")

    with gr.Row():
        dataframe_input = gr.Dataframe(label="数据表格")

    submit_button = gr.Button("提交")

    with gr.Row():
        text_output = gr.Textbox(label="文本输出")
        radio_output = gr.Textbox(label="单选按钮输出")
        checkbox_output = gr.Textbox(label="复选框输出")
        slider_output = gr.Textbox(label="滑块输出")
        file_output = gr.Textbox(label="文件输出")
        image_output = gr.Image(label="图像输出")
        video_output = gr.Video(label="视频输出")
        audio_output = gr.Audio(label="音频输出")
        dataframe_output = gr.Dataframe(label="数据表格输出")

    # 定义按钮点击事件的回调函数
    submit_button.click(
        fn=process_data,
        inputs=[text_input, radio_input, checkbox_input, slider_input, file_input, image_input, video_input,
                audio_input, dataframe_input],
        outputs=[text_output, radio_output, checkbox_output, slider_output, file_output, image_output, video_output,
                 audio_output, dataframe_output]
    )

# 启动界面
demo.launch()

