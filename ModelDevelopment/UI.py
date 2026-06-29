import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.use('TkAgg')


# Color palette
MAIN_BG = "#0f172a"
CARD_BG = "#1e293b"
BTN_BG = "#2563eb"
BTN_HOVER = "#1d4ed8"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
ERROR = "#ef4444"
TEXT = "#f8fafc"
STATUS_BG = "#cbd5e1"


class ImageClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier - CNN Model")
        self.root.geometry("1200x700")
        self.root.configure(bg=MAIN_BG)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.load_model_and_classes()
        self.setup_ui()

    def load_model_and_classes(self):
        try:
            self.model = load_model("..\\ExeProject\\CNN_Model.keras")

            with open("..\\ExeProject\\class_names.pkl", "rb") as f:
                self.class_names = pickle.load(f)

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to load model or classes:\n{str(e)}"
            )
            self.root.quit()

    def setup_ui(self):

        # Main frame
        main_frame = tk.Frame(
            self.root,
            bg=MAIN_BG
        )
        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=20
        )

        # Left panel
        left_panel = tk.Frame(
            main_frame,
            bg=CARD_BG,
            relief=tk.RAISED,
            bd=2
        )
        left_panel.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(0, 10)
        )

        self.image_label = tk.Label(
            left_panel,
            bg=CARD_BG,
            text="No Image Selected",
            font=("Arial", 14),
            fg=TEXT
        )
        self.image_label.pack(
            expand=True,
            fill=tk.BOTH,
            padx=10,
            pady=10
        )

        browse_btn = tk.Button(
            left_panel,
            text="📁 Browse Image",
            command=self.browse_image,
            font=("Arial", 12, "bold"),
            bg=BTN_BG,
            activebackground=BTN_HOVER,
            fg=TEXT,
            padx=20,
            pady=10,
            cursor="hand2",
            relief=tk.FLAT
        )
        browse_btn.pack(pady=20)

        # Right panel
        right_panel = tk.Frame(
            main_frame,
            bg=CARD_BG,
            relief=tk.RAISED,
            bd=2
        )
        right_panel.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True
        )

        title_label = tk.Label(
            right_panel,
            text="Prediction Results",
            font=("Arial", 16, "bold"),
            bg=CARD_BG,
            fg=TEXT
        )
        title_label.pack(pady=15)

        # Top prediction
        top_frame = tk.Frame(
            right_panel,
            bg=CARD_BG
        )
        top_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            top_frame,
            text="Top Prediction:",
            font=("Arial", 14, "bold"),
            bg=CARD_BG,
            fg=WARNING
        ).pack(side=tk.LEFT)

        self.top_pred_label = tk.Label(
            top_frame,
            text="---",
            font=("Arial", 14, "bold"),
            bg=CARD_BG,
            fg=SUCCESS
        )
        self.top_pred_label.pack(side=tk.LEFT, padx=10)

        # Confidence
        conf_frame = tk.Frame(
            right_panel,
            bg=CARD_BG
        )
        conf_frame.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(
            conf_frame,
            text="Confidence:",
            font=("Arial", 12),
            bg=CARD_BG,
            fg=TEXT
        ).pack(side=tk.LEFT)

        self.confidence_label = tk.Label(
            conf_frame,
            text="---",
            font=("Arial", 12, "bold"),
            bg=CARD_BG,
            fg=SUCCESS
        )
        self.confidence_label.pack(side=tk.LEFT, padx=10)

        separator = ttk.Separator(
            right_panel,
            orient="horizontal"
        )
        separator.pack(
            fill=tk.X,
            padx=20,
            pady=15
        )

        # Chart notebook
        self.chart_notebook = ttk.Notebook(right_panel)
        self.chart_notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=10
        )

        # Top 3 chart
        self.top3_frame = tk.Frame(
            self.chart_notebook,
            bg=CARD_BG
        )
        self.chart_notebook.add(
            self.top3_frame,
            text="Top 3"
        )

        self.fig_top3, self.ax_top3 = plt.subplots(
            figsize=(5, 4),
            facecolor=CARD_BG
        )
        self.ax_top3.set_facecolor(CARD_BG)

        self.canvas_top3 = FigureCanvasTkAgg(
            self.fig_top3,
            master=self.top3_frame
        )
        self.canvas_top3.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        # Full chart
        self.full_frame = tk.Frame(
            self.chart_notebook,
            bg=CARD_BG
        )
        self.chart_notebook.add(
            self.full_frame,
            text="All Classes"
        )

        self.fig_full, self.ax_full = plt.subplots(
            figsize=(6, 4),
            facecolor=CARD_BG
        )
        self.ax_full.set_facecolor(CARD_BG)

        self.canvas_full = FigureCanvasTkAgg(
            self.fig_full,
            master=self.full_frame
        )
        self.canvas_full.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 10),
            bg=STATUS_BG
        )
        self.status_bar.pack(
            side=tk.BOTTOM,
            fill=tk.X
        )

        self.update_bar_chart([], [])
        self.update_full_chart([])

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.status_bar.config(
                text=f"Loading: {file_path}"
            )
            self.process_image(file_path)

    def process_image(self, image_path):
        try:
            img = cv2.imread(image_path)

            if img is None:
                messagebox.showerror(
                    "Error",
                    "Could not load image"
                )
                return

            img_rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            self.display_image(img_rgb)

            img_resized = cv2.resize(
                img_rgb,
                (64, 64)
            )

            img_normalized = img_resized / 255.0
            img_input = np.expand_dims(
                img_normalized,
                axis=0
            )

            predictions = self.model.predict(
                img_input,
                verbose=0
            )[0]

            top_3_indices = np.argsort(
                predictions
            )[-3:][::-1]

            top_3_classes = [
                self.class_names[i]
                for i in top_3_indices
            ]

            top_3_confidences = [
                predictions[i]
                for i in top_3_indices
            ]

            self.top_pred_label.config(
                text=top_3_classes[0]
            )

            self.confidence_label.config(
                text=f"{top_3_confidences[0]:.2%}"
            )

            self.update_bar_chart(
                top_3_classes,
                top_3_confidences
            )

            self.update_full_chart(
                predictions
            )

            self.status_bar.config(
                text=f"Predicted: {top_3_classes[0]} ({top_3_confidences[0]:.2%})"
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Prediction failed:\n{str(e)}"
            )
            self.status_bar.config(
                text="Error occurred"
            )

    def display_image(self, image):
        height, width = image.shape[:2]

        max_height = 400
        max_width = 500

        scaling_factor = min(
            max_width / width,
            max_height / height
        )

        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        resized_image = cv2.resize(
            image,
            (new_width, new_height)
        )

        image_pil = Image.fromarray(
            resized_image
        )

        photo = ImageTk.PhotoImage(
            image_pil
        )

        self.image_label.config(
            image=photo,
            text=""
        )

        self.image_label.image = photo

    def update_bar_chart(
        self,
        class_names,
        confidences
    ):
        self.ax_top3.clear()
        self.ax_top3.set_facecolor(CARD_BG)

        if class_names and confidences:
            colors = [
                SUCCESS,
                WARNING,
                ERROR
            ]

            bars = self.ax_top3.barh(
                class_names,
                confidences,
                color=colors[:len(class_names)]
            )

            for bar, conf in zip(
                bars,
                confidences
            ):
                width = bar.get_width()

                self.ax_top3.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{conf:.2%}",
                    va="center",
                    color=TEXT
                )

            self.ax_top3.set_xlim(0, 1)

            self.ax_top3.set_xlabel(
                "Confidence",
                color=TEXT
            )

            self.ax_top3.set_title(
                "Top 3 Predictions",
                color=TEXT
            )

            self.ax_top3.tick_params(
                axis="x",
                colors=TEXT
            )

            self.ax_top3.tick_params(
                axis="y",
                colors=TEXT
            )

        self.canvas_top3.draw()

    def update_full_chart(
        self,
        predictions
    ):
        self.ax_full.clear()
        self.ax_full.set_facecolor(CARD_BG)

        if len(predictions) > 0:
            sorted_indices = np.argsort(
                predictions
            )[::-1]

            class_names = [
                self.class_names[i]
                for i in sorted_indices
            ]

            sorted_probs = [
                predictions[i]
                for i in sorted_indices
            ]

            bars = self.ax_full.bar(
                class_names,
                sorted_probs,
                color="#64748b"
            )

            bars[0].set_color(SUCCESS)

            self.ax_full.set_ylim(0, 1)

            self.ax_full.set_title(
                "All Class Probabilities",
                color=TEXT
            )

            self.ax_full.set_ylabel(
                "Probability",
                color=TEXT
            )

            self.ax_full.tick_params(
                axis="x",
                rotation=45,
                colors=TEXT
            )

            self.ax_full.tick_params(
                axis="y",
                colors=TEXT
            )

            self.fig_full.tight_layout()

        self.canvas_full.draw()

    def on_closing(self):
        try:
            plt.close("all")
            self.root.quit()
            self.root.destroy()
        except:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierUI(root)
    root.mainloop()