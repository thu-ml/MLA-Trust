import base64
import io
import os

from jinja2 import Template
from PIL import Image

template_str = """<!DOCTYPE html>
<html>
<head>
    <title>Task Execution Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }
        .example {
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 30px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .task {
            background-color: #f0f7ff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .task h3 {
            margin-top: 0;
            color: #0066cc;
        }
        .important_notes {
            background-color: #fff4f0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .important_notes h3 {
            margin-top: 0;
            color: #cc5500;
        }
        .finish_thought {
            background-color: #f5f0ff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .finish_thought h3 {
            margin-top: 0;
            color: #5500cc;
        }
        .action-history {
            background-color: #fff8e6;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .action-history h3 {
            margin-top: 0;
            color: #cc7700;
        }
        .action-item {
            margin-bottom: 5px;
            padding-left: 10px;
            border-left: 3px solid #ff9900;
        }
        .refuse_flag {
            background-color: #ffeeee;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .refuse_flag h3 {
            margin-top: 0;
            color: #cc0000;
        }
        .screenshot {
            background-color: #f0f0f7;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .screenshot h3 {
            margin-top: 0;
            color: #4a4a8c;
        }
        .screenshot img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin: 5px 0;
        }
        .answer {
            background-color: #f0fff0;
            padding: 10px;
            border-radius: 5px;
        }
        .answer h3 {
            margin-top: 0;
            color: #007700;
        }
        .metrics {
            background-color: #f0f0ff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .metrics h3 {
            margin-top: 0;
            color: #4444cc;
        }
        .metrics table {
            width: 100%;
            border-collapse: collapse;
        }
        .metrics th, .metrics td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .metrics th {
            background-color: #e6e6ff;
        }
        h1 {
            color: #1a5dad;
            border-bottom: 3px solid #3e7dcc;
            padding-bottom: 12px;
            font-size: 28px;
            margin-bottom: 25px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }

        h2 {
            color: #0c3b7d;
            font-size: 22px;
            border-left: 4px solid #5990d5;
            padding-left: 10px;
            margin-top: 20px;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <h1>Task Execution Report</h1>
    
    {% for example in examples %}
    <div class="example">
        <h2>Example {{ loop.index0 }}</h2>
        
        <div class="task">
            <h3>Task</h3>
            <p>{{ example.task }}</p>
        </div>
        
        <div class="action-history">
            <h3>Action History</h3>
            <ul>
                {% for action in example.action_history %}
                <li class="action-item">{{ action }}</li>
                {% endfor %}
            </ul>
        </div>
        
        <div class="screenshot">
            <h3>Screenshot</h3>
            <img src="{{ example.screenshot }}" alt="Screenshot" width="300" height="auto">
        </div>

        <div class="important_notes">
            <h3>Important Notes</h3>
            <p>{{ example.important_notes }}</p>
        </div>

        <div class="finish_thought">
            <h3>Finish Thought</h3>
            <p>{{ example.finish_thought }}</p>
        </div>
        
        <div class="refuse_flag">
            <h3>Refuse Flag</h3>
            <p>{{ example.refuse_flag }}</p>
        </div>

        <div class="metrics">
            <h3>Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                {% for key, value in example.metrics.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>

        <div class="answer">
            <h3>Answer</h3>
            <p>{{ example.ans }}</p>
        </div>
    </div>
    {% endfor %}
</body>
</html>"""


template = Template(template_str)


def image_to_base64(image_path):
    if image_path is None or not os.path.exists(image_path):
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    with open(image_path, "rb") as img_file:
        # resize the image to save space
        width = 250
        img = Image.open(img_file)
        w_percent = width / float(img.size[0])
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((width, h_size))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        base64_image = base64.b64encode(img_byte_arr).decode("utf-8")

    img_format = image_path.lower().split(".")[-1]
    if img_format in ["jpg", "jpeg"]:
        img_format = "jpeg"
    elif img_format == "png":
        img_format = "png"
    else:
        img_format = "jpeg"

    return f"data:image/{img_format};base64,{base64_image}"
