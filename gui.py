"""
GUI Tkinter para el proyecto "clasificador_piezas_ia_2025" (versión mejorada)

Cambios vs v1:
- Debug contornos: filtrar_contornos() devuelve (contornos, hierarchy) => se corrige unpack.
- Debug imágenes: Labels sin width/height fijos (en Tk son "unidades de texto", no pixeles).
- Audio: ejecución de comandos SIN imprimir en consola:
    - contar: muestra conteo en la interfaz.
    - proporcion: muestra posterior y actualiza la pestaña Bayes.
    - salir: genera resumen y cierra (con confirmación).
- Resumen: ahora lista comandos ejecutados + resultados guardados.
"""

import os
import re
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional, Dict, List, Tuple, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import cv2
except Exception:
    cv2 = None

# ===== Imports del proyecto =====
try:
    from src.predecir_k4_manual import predecir_k4
except Exception:
    from predecir_k4_manual import predecir_k4  # fallback

try:
    import agente as agente_backend
except Exception:
    import agente as agente_backend  # si está en raíz

try:
    from src.bayes import estimar_posterior_secuencial
except Exception:
    from bayes import estimar_posterior_secuencial

try:
    from src.binario import binarizar
except Exception:
    from binario import binarizar

try:
    from src.contornos import filtrar_contornos
except Exception:
    from contornos import filtrar_contornos


CARPETA_MUESTRA_IMG = "data/muestras"
CARPETA_MUESTRA_AUDIO = "data_voz/muestras_audio"


def _extraer_numero(nombre_archivo: str) -> Optional[int]:
    base = os.path.splitext(nombre_archivo)[0]
    m = re.match(r"^\s*(\d+)", base)
    return int(m.group(1)) if m else None


def listar_wavs_ordenados(carpeta: str) -> List[str]:
    if not os.path.isdir(carpeta):
        return []
    wavs = [f for f in os.listdir(carpeta) if f.lower().endswith(".wav")]
    con_num, sin_num = [], []
    for f in wavs:
        n = _extraer_numero(f)
        if n is None:
            sin_num.append(f)
        else:
            con_num.append((n, f))
    con_num.sort(key=lambda t: t[0])
    sin_num.sort()
    ordenados = [f for _, f in con_num] + sin_num
    return [os.path.join(carpeta, f) for f in ordenados]


def fit_image(pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = pil_img.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    nw, nh = int(w * scale), int(h * scale)
    if nw <= 0 or nh <= 0:
        return pil_img
    return pil_img.resize((nw, nh), Image.Resampling.LANCZOS)


def pil_from_cv(img_bgr: np.ndarray) -> Image.Image:
    if img_bgr.ndim == 2:
        return Image.fromarray(img_bgr)
    rgb = img_bgr[..., ::-1]
    return Image.fromarray(rgb)


@dataclass
class AppState:
    piezas: List[str] = field(default_factory=list)
    img_results: List[Tuple[str, str]] = field(default_factory=list)  # (ruta, clase)

    selected_audio: Optional[str] = None
    last_command: Optional[str] = None
    last_proba: Optional[Dict[str, float]] = None
    command_history: List[Dict[str, Any]] = field(default_factory=list)

    # resultados por comando (para mostrar en UI y resumen)
    last_action_result: str = ""
    action_results: List[str] = field(default_factory=list)

    posterior: Optional[Dict[str, float]] = None
    historial: Optional[List[Dict[str, float]]] = None

    selected_image: Optional[str] = None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clasificador de Piezas IA — GUI (Tkinter)")
        self.geometry("1200x780")
        self.minsize(1000, 700)

        self.state = AppState()
        self._imgtk_cache: List[ImageTk.PhotoImage] = []

        self._build_ui()
        self._refresh_audio_list()
        self._refresh_image_list_for_debug()

    # ---------------- UI base ----------------
    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=12, pady=10)

        self.lbl_status = ttk.Label(header, text="Estado: listo.", font=("Segoe UI", 11))
        self.lbl_status.pack(side="left")

        self.lbl_lastcmd = ttk.Label(header, text="Último comando: —", font=("Segoe UI", 11))
        self.lbl_lastcmd.pack(side="left", padx=16)

        ttk.Button(header, text="Salir", command=self.on_exit).pack(side="right")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=12, pady=10)

        self.tab_imgs = ttk.Frame(self.nb)
        self.tab_debug = ttk.Frame(self.nb)
        self.tab_audio = ttk.Frame(self.nb)
        self.tab_bayes = ttk.Frame(self.nb)
        self.tab_resumen = ttk.Frame(self.nb)

        self.nb.add(self.tab_imgs, text="Imágenes")
        self.nb.add(self.tab_debug, text="Debug visión")
        self.nb.add(self.tab_audio, text="Audio")
        self.nb.add(self.tab_bayes, text="Bayes / Proporción")
        self.nb.add(self.tab_resumen, text="Resumen")

        self._build_tab_imgs()
        self._build_tab_debug()
        self._build_tab_audio()
        self._build_tab_bayes()
        self._build_tab_resumen()

    def set_status(self, text: str):
        self.lbl_status.configure(text=f"Estado: {text}")
        self.update_idletasks()

    def set_last_command(self, cmd: Optional[str]):
        self.lbl_lastcmd.configure(text=f"Último comando: {cmd or '—'}")
        self.update_idletasks()

    # ---------------- TAB IMÁGENES ----------------
    def _build_tab_imgs(self):
        top = ttk.Frame(self.tab_imgs)
        top.pack(fill="x", pady=8, padx=8)

        ttk.Label(top, text="Carpeta:").pack(side="left")
        self.var_img_folder = tk.StringVar(value=CARPETA_MUESTRA_IMG)
        ttk.Entry(top, textvariable=self.var_img_folder, width=45).pack(side="left", padx=6)
        ttk.Button(top, text="Elegir...", command=self.choose_img_folder).pack(side="left")
        ttk.Button(top, text="Reconocer imágenes", command=self.run_image_recognition).pack(side="left", padx=10)

        mid = ttk.Frame(self.tab_imgs)
        mid.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True)

        self.canvas_gallery = tk.Canvas(left, bg="#0f0f0f", highlightthickness=0)
        self.scroll_gallery = ttk.Scrollbar(left, orient="vertical", command=self.canvas_gallery.yview)
        self.canvas_gallery.configure(yscrollcommand=self.scroll_gallery.set)

        self.frame_gallery = ttk.Frame(self.canvas_gallery)
        self.canvas_gallery.create_window((0, 0), window=self.frame_gallery, anchor="nw")

        self.canvas_gallery.pack(side="left", fill="both", expand=True)
        self.scroll_gallery.pack(side="right", fill="y")

        self.frame_gallery.bind("<Configure>", lambda e: self.canvas_gallery.configure(scrollregion=self.canvas_gallery.bbox("all")))

        right = ttk.Frame(mid, width=260)
        right.pack(side="right", fill="y", padx=(10, 0))

        ttk.Label(right, text="Conteo por clase", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self.txt_counts = tk.Text(right, height=18, width=30)
        self.txt_counts.pack(fill="y", expand=False)

    def choose_img_folder(self):
        folder = filedialog.askdirectory(title="Elegir carpeta de imágenes", initialdir="data")
        if folder:
            self.var_img_folder.set(folder)
            self._refresh_image_list_for_debug()

    def run_image_recognition(self):
        folder = self.var_img_folder.get().strip()
        if not os.path.isdir(folder):
            messagebox.showerror("Error", f"No existe la carpeta: {folder}")
            return

        self.set_status("clasificando imágenes…")
        self.state.piezas.clear()
        self.state.img_results.clear()
        self._imgtk_cache.clear()

        imgs = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not imgs:
            messagebox.showinfo("Sin imágenes", "No se encontraron imágenes.")
            self.set_status("sin imágenes")
            return

        for name in imgs:
            ruta = os.path.join(folder, name)
            try:
                clase, _ = predecir_k4(ruta)
                clase = str(clase).strip().lower()
            except Exception as e:
                clase = f"error: {e}"
            self.state.img_results.append((ruta, clase))
            if not clase.startswith("error"):
                self.state.piezas.append(clase)

        self._render_gallery()
        self._render_counts()
        self._refresh_image_list_for_debug()
        self.set_status(f"imágenes clasificadas: {len(self.state.img_results)}")

    def _render_gallery(self):
        for w in self.frame_gallery.winfo_children():
            w.destroy()

        cols = 4
        thumb_w, thumb_h = 220, 170

        for idx, (ruta, clase) in enumerate(self.state.img_results):
            r = idx // cols
            c = idx % cols

            card = ttk.Frame(self.frame_gallery, padding=8, relief="ridge")
            card.grid(row=r, column=c, padx=8, pady=8, sticky="nsew")

            try:
                pil_img = Image.open(ruta).convert("RGB")
                pil_thumb = fit_image(pil_img, thumb_w, thumb_h)
                imgtk = ImageTk.PhotoImage(pil_thumb)
                self._imgtk_cache.append(imgtk)
                ttk.Label(card, image=imgtk).pack()
            except Exception:
                ttk.Label(card, text="[No se pudo cargar]").pack()

            ttk.Label(card, text=f"{os.path.basename(ruta)}\n→ {clase}", justify="center").pack(pady=(6, 0))

            def _on_click(path=ruta):
                self.nb.select(self.tab_debug)
                self.state.selected_image = path
                self.var_debug_img.set(path)
                self._render_debug_original(path)

            card.bind("<Button-1>", lambda e, p=ruta: _on_click(p))
            for child in card.winfo_children():
                child.bind("<Button-1>", lambda e, p=ruta: _on_click(p))

    def _render_counts(self):
        self.txt_counts.delete("1.0", "end")
        if not self.state.piezas:
            self.txt_counts.insert("end", "Aún no hay piezas.\n")
            return
        conteo = Counter(self.state.piezas)
        total = sum(conteo.values())
        self.txt_counts.insert("end", f"Total piezas: {total}\n\n")
        for k, v in conteo.most_common():
            self.txt_counts.insert("end", f"- {k}: {v}\n")

    # ---------------- TAB DEBUG VISIÓN ----------------
    def _build_tab_debug(self):
        outer = ttk.Frame(self.tab_debug)
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        top = ttk.Frame(outer)
        top.pack(fill="x")

        ttk.Label(top, text="Imagen:").pack(side="left")
        self.var_debug_img = tk.StringVar(value="")
        self.cmb_debug_img = ttk.Combobox(top, textvariable=self.var_debug_img, state="readonly", width=80)
        self.cmb_debug_img.pack(side="left", padx=6)

        ttk.Button(top, text="Elegir archivo...", command=self.choose_single_image_for_debug).pack(side="left")
        ttk.Button(top, text="Ver binario", command=self.show_binary).pack(side="left", padx=8)
        ttk.Button(top, text="Ver contornos", command=self.show_contours).pack(side="left")

        panels = ttk.Frame(outer)
        panels.pack(fill="both", expand=True, pady=10)

        ttk.Label(panels, text="Original").grid(row=0, column=0, sticky="w")
        ttk.Label(panels, text="Binario").grid(row=0, column=1, sticky="w")
        ttk.Label(panels, text="Contornos").grid(row=0, column=2, sticky="w")

        # Labels sin width/height: que el widget se adapte al tamaño de la imagen
        self.can_orig = tk.Label(panels, bg="#111")
        self.can_bin = tk.Label(panels, bg="#111")
        self.can_cnt = tk.Label(panels, bg="#111")

        self.can_orig.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.can_bin.grid(row=1, column=1, sticky="nsew", padx=(0, 8))
        self.can_cnt.grid(row=1, column=2, sticky="nsew")

        for i in range(3):
            panels.columnconfigure(i, weight=1)
        panels.rowconfigure(1, weight=1)

    def choose_single_image_for_debug(self):
        filetypes = [("Imágenes", "*.png *.jpg *.jpeg"), ("Todos", "*.*")]
        path = filedialog.askopenfilename(title="Elegir imagen", filetypes=filetypes)
        if path:
            self.state.selected_image = path
            self.var_debug_img.set(path)
            self._render_debug_original(path)

    def _refresh_image_list_for_debug(self):
        folder = self.var_img_folder.get().strip()
        paths = []
        if os.path.isdir(folder):
            imgs = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            paths = [os.path.join(folder, f) for f in imgs]
        self.cmb_debug_img["values"] = paths
        if paths and not self.var_debug_img.get():
            self.var_debug_img.set(paths[0])
            self.state.selected_image = paths[0]
            self._render_debug_original(paths[0])

    def _get_debug_image_path(self) -> Optional[str]:
        path = self.var_debug_img.get().strip() or self.state.selected_image
        return path if path and os.path.exists(path) else None

    def _render_debug_original(self, path: str):
        if cv2 is None:
            messagebox.showwarning("Aviso", "OpenCV no está disponible. Debug limitado.")
            return
        try:
            pil_img = Image.open(path).convert("RGB")
            pil_fit = fit_image(pil_img, 360, 360)
            imgtk = ImageTk.PhotoImage(pil_fit)
            self._imgtk_cache.append(imgtk)
            self.can_orig.configure(image=imgtk)
            self.can_orig.image = imgtk

            self.can_bin.configure(image="")
            self.can_cnt.configure(image="")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo mostrar la imagen: {e}")

    def show_binary(self):
        if binarizar is None or cv2 is None:
            messagebox.showerror("Error", "Necesitás binarizar() y cv2 para este debug.")
            return
        path = self._get_debug_image_path()
        if not path:
            messagebox.showinfo("Info", "Elegí una imagen primero.")
            return
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imread devolvió None.")
            mask = binarizar(img)
            pil_mask = Image.fromarray(mask).convert("L")
            pil_fit = fit_image(pil_mask, 360, 360)
            imgtk = ImageTk.PhotoImage(pil_fit)
            self._imgtk_cache.append(imgtk)
            self.can_bin.configure(image=imgtk)
            self.can_bin.image = imgtk
            self.set_status("binario mostrado")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar binario: {e}")

    def show_contours(self):
        if filtrar_contornos is None or binarizar is None or cv2 is None:
            messagebox.showerror("Error", "Necesitás filtrar_contornos(), binarizar() y cv2.")
            return

        path = self._get_debug_image_path()
        if not path:
            messagebox.showinfo("Info", "Elegí una imagen primero.")
            return

        try:
        # 1) Leer imagen original
            img_original = cv2.imread(path, cv2.IMREAD_COLOR)
            if img_original is None:
                raise ValueError("cv2.imread devolvió None.")

        # 2) Obtener binario (esto YA está redimensionado, ej. 512x512)
            mask = binarizar(img_original)

        # 3) Obtener contornos (en coordenadas del binario)
            contornos, _ = filtrar_contornos(mask)

        # 4) Crear imagen base para dibujar: versión redimensionada
            img_resized = cv2.resize(
                img_original,
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_AREA
            )

        # 5) Dibujar contornos correctamente alineados
            cv2.drawContours(img_resized, contornos, -1, (0, 0, 255), 2)

        # 6) Mostrar en GUI
            pil_draw = fit_image(pil_from_cv(img_resized), 360, 360)
            imgtk = ImageTk.PhotoImage(pil_draw)
            self._imgtk_cache.append(imgtk)
            self.can_cnt.configure(image=imgtk)
            self.can_cnt.image = imgtk

            self.set_status("contornos mostrados")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron mostrar contornos: {e}")


    # ---------------- TAB AUDIO ----------------
    def _build_tab_audio(self):
        outer = ttk.Frame(self.tab_audio)
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, 6))

        ttk.Label(top, text="Carpeta audios:").pack(side="left")
        self.var_audio_folder = tk.StringVar(value=CARPETA_MUESTRA_AUDIO)
        ttk.Entry(top, textvariable=self.var_audio_folder, width=45).pack(side="left", padx=6)
        ttk.Button(top, text="Elegir...", command=self.choose_audio_folder).pack(side="left")
        ttk.Button(top, text="Refrescar", command=self._refresh_audio_list).pack(side="left", padx=8)

        mid = ttk.Frame(outer)
        mid.pack(fill="both", expand=True)

        left = ttk.Frame(mid, width=320)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Audios (.wav)", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.lst_audios = tk.Listbox(left, height=18)
        self.lst_audios.pack(fill="y", expand=True, pady=6)
        self.lst_audios.bind("<<ListboxSelect>>", self.on_select_audio)

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Reconocer", command=self.run_audio_predict).pack(side="left")
        ttk.Button(btns, text="Confirmar", command=self.confirm_and_execute).pack(side="left", padx=8)
        ttk.Button(btns, text="Cancelar", command=self.cancel_command).pack(side="left")

        right = ttk.Frame(mid)
        right.pack(side="right", fill="both", expand=True, padx=(12, 0))

        ttk.Label(right, text="Resultado", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.lbl_cmd = ttk.Label(right, text="Comando: —", font=("Segoe UI", 13))
        self.lbl_cmd.pack(anchor="w", pady=(6, 6))

        ttk.Label(right, text="Probabilidades:").pack(anchor="w")
        self.txt_proba = tk.Text(right, height=6)
        self.txt_proba.pack(fill="x", pady=6)

        ttk.Label(right, text="Salida del comando (en la GUI):").pack(anchor="w")
        self.txt_action = tk.Text(right, height=10)
        self.txt_action.pack(fill="both", expand=True, pady=(0, 6))

        ttk.Label(right, text="Historial de comandos:").pack(anchor="w")
        self.txt_hist = tk.Text(right, height=7)
        self.txt_hist.pack(fill="x")

    def choose_audio_folder(self):
        folder = filedialog.askdirectory(title="Elegir carpeta de audios", initialdir="data_voz")
        if folder:
            self.var_audio_folder.set(folder)
            self._refresh_audio_list()

    def _refresh_audio_list(self):
        folder = self.var_audio_folder.get().strip()
        paths = listar_wavs_ordenados(folder)
        self.lst_audios.delete(0, "end")
        for p in paths:
            self.lst_audios.insert("end", os.path.basename(p))
        self._audio_paths = paths
        self.state.selected_audio = paths[0] if paths else None
        self.set_status("lista audios cargada" if paths else "sin audios")

    def on_select_audio(self, _evt=None):
        sel = self.lst_audios.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(getattr(self, "_audio_paths", [])):
            self.state.selected_audio = self._audio_paths[idx]
            self.set_status(f"audio: {os.path.basename(self.state.selected_audio)}")

    def run_audio_predict(self):
        audio_path = self.state.selected_audio
        if not audio_path or not os.path.exists(audio_path):
            messagebox.showinfo("Info", "Elegí un audio válido.")
            return

        self.set_status("reconociendo comando…")
        try:
            cmd, proba = agente_backend.predecir_comando_desde_wav(audio_path)
        except Exception as e:
            messagebox.showerror("Error", f"Falló el reconocimiento: {e}")
            self.set_status("error")
            return

        self.state.last_command = cmd
        self.state.last_proba = proba
        self.set_last_command(cmd)

        self.lbl_cmd.configure(text=f"Comando: {cmd}")
        self.txt_proba.delete("1.0", "end")
        if proba:
            for cls, p in sorted(proba.items(), key=lambda t: t[1], reverse=True):
                self.txt_proba.insert("end", f"{cls}: {p:.4f}\n")
        else:
            self.txt_proba.insert("end", "(sin probabilidades)\n")

        self.set_status("comando reconocido (confirmar para ejecutar)")

    def _accion_contar(self) -> str:
        if not self.state.piezas:
            return "No hay piezas para contar.\n"
        conteo = Counter(self.state.piezas)
        lines = ["=== CONTEO DE PIEZAS ===", f"Total: {sum(conteo.values())}"]
        for k, v in conteo.most_common():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines) + "\n"

    def _accion_proporcion(self) -> str:
        if not self.state.piezas:
            return "No hay piezas para aplicar Bayes.\n"
        posterior, historial = estimar_posterior_secuencial(self.state.piezas)
        self.state.posterior = posterior
        self.state.historial = historial
        # actualizar plots
        self._plot_bayes_bar(posterior)
        self._plot_bayes_history(historial)

        lines = ["=== BAYES (PROPORCIÓN) ==="]
        for caja, p in posterior.items():
            lines.append(f"- Caja {caja}: {p:.4f}")
        caja_mas = max(posterior, key=posterior.get)
        lines.append(f"Caja más probable: {caja_mas}")
        return "\n".join(lines) + "\n"

    def confirm_and_execute(self):
        cmd = self.state.last_command
        if not cmd:
            messagebox.showinfo("Info", "Primero reconocé un comando.")
            return

        if not messagebox.askyesno("Confirmar", f"¿Ejecutar comando '{cmd}'?"):
            return

        # Ejecutamos en modo GUI (sin prints en consola)
        if cmd == "contar":
            out = self._accion_contar()
            terminar = False
        elif cmd == "proporcion":
            out = self._accion_proporcion()
            terminar = False
            # Nos movemos a la pestaña Bayes para ver el gráfico
            self.nb.select(self.tab_bayes)
        elif cmd == "salir":
            out = "Comando 'salir' recibido. Se generará el resumen.\n"
            terminar = True
        else:
            out = f"Comando no reconocido en GUI: {cmd}\n"
            terminar = False

        self.state.last_action_result = out
        self.state.action_results.append(out)

        self.txt_action.delete("1.0", "end")
        self.txt_action.insert("end", out)

        self.state.command_history.append({
            "audio": os.path.basename(self.state.selected_audio or ""),
            "comando": cmd,
            "ejecutado": True
        })
        self._render_history()

        self.set_status(f"comando ejecutado: {cmd}")

        if terminar:
            self.generate_summary()
            self.on_exit()

    def cancel_command(self):
        if self.state.last_command:
            self.state.command_history.append({
                "audio": os.path.basename(self.state.selected_audio or ""),
                "comando": self.state.last_command,
                "ejecutado": False
            })
        self.state.last_command = None
        self.state.last_proba = None
        self.set_last_command(None)

        self.lbl_cmd.configure(text="Comando: —")
        self.txt_proba.delete("1.0", "end")
        self.txt_action.delete("1.0", "end")
        self.txt_action.insert("end", "(cancelado)\n")

        self._render_history()
        self.set_status("comando cancelado")

    def _render_history(self):
        self.txt_hist.delete("1.0", "end")
        if not self.state.command_history:
            self.txt_hist.insert("end", "Sin historial.\n")
            return
        for i, e in enumerate(self.state.command_history, start=1):
            ok = "OK" if e.get("ejecutado") else "CANCELADO"
            self.txt_hist.insert("end", f"{i:02d}) {ok} {e.get('audio')} → {e.get('comando')}\n")

    # ---------------- TAB BAYES ----------------
    def _build_tab_bayes(self):
        outer = ttk.Frame(self.tab_bayes)
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, 6))

        ttk.Button(top, text="Actualizar Bayes", command=self.update_bayes_from_current_pieces).pack(side="left")
        ttk.Button(top, text="Limpiar", command=self.clear_bayes_plots).pack(side="left", padx=8)

        plots = ttk.Frame(outer)
        plots.pack(fill="both", expand=True)

        self.fig_bar = Figure(figsize=(5, 4), dpi=100)
        self.ax_bar = self.fig_bar.add_subplot(111)
        self.ax_bar.set_title("Probabilidad por caja (posterior)")

        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=plots)
        self.canvas_bar.get_tk_widget().pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.fig_line = Figure(figsize=(5, 4), dpi=100)
        self.ax_line = self.fig_line.add_subplot(111)
        self.ax_line.set_title("Evolución (historial)")

        self.canvas_line = FigureCanvasTkAgg(self.fig_line, master=plots)
        self.canvas_line.get_tk_widget().pack(side="right", fill="both", expand=True)

    def update_bayes_from_current_pieces(self):
        if not self.state.piezas:
            messagebox.showinfo("Info", "No hay piezas clasificadas.")
            return
        posterior, historial = estimar_posterior_secuencial(self.state.piezas)
        self.state.posterior = posterior
        self.state.historial = historial
        self._plot_bayes_bar(posterior)
        self._plot_bayes_history(historial)
        self.set_status("bayes actualizado")

    def clear_bayes_plots(self):
        self.ax_bar.clear()
        self.ax_line.clear()
        self.canvas_bar.draw()
        self.canvas_line.draw()

    def _plot_bayes_bar(self, posterior: Dict[str, float]):
        self.ax_bar.clear()
        cajas = list(posterior.keys())
        vals = [posterior[c] for c in cajas]
        self.ax_bar.bar(cajas, vals)
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_title("Probabilidad por caja (posterior)")
        self.canvas_bar.draw()

    def _plot_bayes_history(self, historial: List[Dict[str, float]]):
        self.ax_line.clear()
        cajas = list(historial[0].keys())
        x = list(range(len(historial)))
        for caja in cajas:
            y = [h.get(caja, 0.0) for h in historial]
            self.ax_line.plot(x, y, label=f"Caja {caja}")
        self.ax_line.set_ylim(0, 1)
        self.ax_line.legend(loc="best")
        self.ax_line.set_title("Evolución (historial)")
        self.canvas_line.draw()

    # ---------------- TAB RESUMEN ----------------
    def _build_tab_resumen(self):
        outer = ttk.Frame(self.tab_resumen)
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        ttk.Button(outer, text="Generar resumen", command=self.generate_summary).pack(anchor="w")
        self.txt_summary = tk.Text(outer)
        self.txt_summary.pack(fill="both", expand=True, pady=8)

    def generate_summary(self):
        self.txt_summary.delete("1.0", "end")
        self.txt_summary.insert("end", "=== RESUMEN ===\n\n")

        # Piezas
        if self.state.piezas:
            conteo = Counter(self.state.piezas)
            self.txt_summary.insert("end", f"Total piezas: {sum(conteo.values())}\n")
            for k, v in conteo.most_common():
                self.txt_summary.insert("end", f"- {k}: {v}\n")
        else:
            self.txt_summary.insert("end", "No hay piezas clasificadas.\n")

        # Bayes
        self.txt_summary.insert("end", "\n=== BAYES ===\n")
        if self.state.posterior:
            for caja, p in self.state.posterior.items():
                self.txt_summary.insert("end", f"- Caja {caja}: {p:.4f}\n")
            caja_mas_prob = max(self.state.posterior, key=self.state.posterior.get)
            self.txt_summary.insert("end", f"Caja más probable: {caja_mas_prob}\n")
        else:
            self.txt_summary.insert("end", "Bayes aún no calculado.\n")

        # Comandos
        self.txt_summary.insert("end", "\n=== COMANDOS ===\n")
        if self.state.command_history:
            for i, e in enumerate(self.state.command_history, start=1):
                ok = "EJECUTADO" if e.get("ejecutado") else "CANCELADO"
                self.txt_summary.insert("end", f"{i:02d}) {ok}: {e.get('comando')} ({e.get('audio')})\n")
        else:
            self.txt_summary.insert("end", "Sin comandos.\n")

        # Resultados
        self.txt_summary.insert("end", "\n=== SALIDA / RESULTADOS ===\n")
        if self.state.action_results:
            for i, block in enumerate(self.state.action_results, start=1):
                self.txt_summary.insert("end", f"\n--- Resultado {i} ---\n{block}")
        else:
            self.txt_summary.insert("end", "No hay salidas.\n")

        self.set_status("resumen generado")

    def on_exit(self):
        if messagebox.askokcancel("Salir", "¿Cerrar la aplicación?"):
            self.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
