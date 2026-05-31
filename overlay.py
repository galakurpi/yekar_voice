#!/usr/bin/env python3
import math
import os
import sys
import time
import cairo

os.environ.setdefault("GSK_RENDERER", "cairo")

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Gdk, GLib, Gtk, Pango, PangoCairo


WIDTH = 336
HEIGHT = 78


def rounded_rect(ctx, x, y, width, height, radius):
    ctx.new_sub_path()
    ctx.arc(x + width - radius, y + radius, radius, -math.pi / 2, 0)
    ctx.arc(x + width - radius, y + height - radius, radius, 0, math.pi / 2)
    ctx.arc(x + radius, y + height - radius, radius, math.pi / 2, math.pi)
    ctx.arc(x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def state_color(title):
    normalized = title.strip().lower()
    if normalized in {"pasted", "done"}:
        return (0.28, 0.86, 0.50)
    if normalized in {"failed"}:
        return (1.00, 0.33, 0.33)
    if normalized in {"finalizing", "retry", "no speech"}:
        return (1.00, 0.68, 0.26)
    if normalized in {"ignored", "cancelled"}:
        return (0.58, 0.62, 0.68)
    return (1.00, 0.24, 0.30)


class Bubble(Gtk.Window):
    def __init__(self, title, body, timeout_secs):
        super().__init__(type=Gtk.WindowType.POPUP)
        self.title_text = title
        self.body_text = body
        self.timeout_secs = timeout_secs
        self.started_at = time.monotonic()
        self.color = state_color(title)

        self.set_decorated(False)
        self.set_resizable(False)
        self.set_app_paintable(True)
        self.set_keep_above(True)
        self.set_accept_focus(False)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_type_hint(Gdk.WindowTypeHint.NOTIFICATION)

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual is not None:
            self.set_visual(visual)

        self.canvas = Gtk.DrawingArea()
        self.canvas.set_size_request(WIDTH, HEIGHT)
        self.canvas.connect("draw", self.draw)
        self.add(self.canvas)

        GLib.timeout_add(42, self.tick)
        GLib.idle_add(self.place)
        if timeout_secs:
            GLib.timeout_add_seconds(timeout_secs, Gtk.main_quit)

    def place(self):
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor() or display.get_monitor(0)
        geometry = monitor.get_geometry()
        x = geometry.x + max(0, (geometry.width - WIDTH) // 2)
        y = geometry.y + 34
        self.move(x, y)
        return False

    def tick(self):
        self.canvas.queue_draw()
        return True

    def draw_text(self, ctx, text, x, y, size, weight, alpha):
        layout = PangoCairo.create_layout(ctx)
        layout.set_text(text, -1)
        layout.set_width(176 * Pango.SCALE)
        layout.set_ellipsize(Pango.EllipsizeMode.END)
        desc = Pango.FontDescription()
        desc.set_family("Cantarell")
        desc.set_size(size * Pango.SCALE)
        desc.set_weight(weight)
        layout.set_font_description(desc)
        ctx.set_source_rgba(0.06, 0.08, 0.12, alpha)
        ctx.move_to(x, y)
        PangoCairo.show_layout(ctx, layout)

    def draw_waveform(self, ctx, elapsed):
        x0 = 248
        baseline = 39
        for i in range(7):
            phase = elapsed * 5.5 + i * 0.72
            height = 9 + (math.sin(phase) + 1) * 9
            x = x0 + i * 8
            rounded_rect(ctx, x, baseline - height / 2, 4, height, 2)
            ctx.set_source_rgba(*self.color, 0.88 - i * 0.04)
            ctx.fill()

    def draw_processing(self, ctx, elapsed):
        x0 = 254
        for i in range(3):
            pulse = (math.sin(elapsed * 5.0 + i * 1.2) + 1) / 2
            radius = 3 + pulse * 1.8
            ctx.arc(x0 + i * 14, 39, radius, 0, 2 * math.pi)
            ctx.set_source_rgba(*self.color, 0.42 + pulse * 0.48)
            ctx.fill()

    def draw(self, _widget, ctx):
        elapsed = time.monotonic() - self.started_at
        ctx.set_operator(1)
        ctx.set_source_rgba(0, 0, 0, 0)
        ctx.paint()

        rounded_rect(ctx, 10, 12, WIDTH - 20, HEIGHT - 21, 29)
        ctx.set_source_rgba(0.02, 0.03, 0.05, 0.24)
        ctx.fill()

        rounded_rect(ctx, 8, 8, WIDTH - 16, HEIGHT - 20, 29)
        gradient = cairo.LinearGradient(8, 8, 8, HEIGHT - 12)
        gradient.add_color_stop_rgba(0.00, 1.00, 1.00, 1.00, 0.70)
        gradient.add_color_stop_rgba(0.46, 0.92, 0.96, 1.00, 0.36)
        gradient.add_color_stop_rgba(1.00, 0.74, 0.82, 0.96, 0.25)
        ctx.set_source(gradient)
        ctx.fill_preserve()
        ctx.set_source_rgba(1, 1, 1, 0.72)
        ctx.set_line_width(1.0)
        ctx.stroke()

        rounded_rect(ctx, 14, 13, WIDTH - 28, 18, 14)
        ctx.set_source_rgba(1, 1, 1, 0.34)
        ctx.fill()

        rounded_rect(ctx, 11, 10, WIDTH - 22, HEIGHT - 24, 27)
        ctx.set_source_rgba(0.16, 0.22, 0.32, 0.12)
        ctx.set_line_width(1.0)
        ctx.stroke()

        pulse = (math.sin(elapsed * 4.8) + 1) / 2
        ctx.arc(44, 39, 18 + pulse * 5, 0, 2 * math.pi)
        ctx.set_source_rgba(1, 1, 1, 0.38)
        ctx.fill()
        ctx.arc(44, 39, 15 + pulse * 4, 0, 2 * math.pi)
        ctx.set_source_rgba(*self.color, 0.11 + pulse * 0.16)
        ctx.fill()
        ctx.arc(44, 39, 8, 0, 2 * math.pi)
        ctx.set_source_rgba(*self.color, 0.95)
        ctx.fill()

        title = self.title_text if self.title_text else "Listening"
        if self.body_text:
            subtitle = self.body_text
        elif title.lower() == "listening":
            subtitle = "Alt+Space"
        elif title.lower() == "finalizing":
            subtitle = "Processing"
        else:
            subtitle = ""
        self.draw_text(ctx, title, 74, 19, 13, Pango.Weight.SEMIBOLD, 0.92)
        self.draw_text(ctx, subtitle, 74, 41, 10, Pango.Weight.NORMAL, 0.58)

        if title.lower() == "listening":
            self.draw_waveform(ctx, elapsed)
        elif title.lower() == "finalizing":
            self.draw_processing(ctx, elapsed)
        else:
            ctx.arc(278, 39, 5, 0, 2 * math.pi)
            ctx.set_source_rgba(*self.color, 0.95)
            ctx.fill()

        return False


def main():
    _mode = sys.argv[1] if len(sys.argv) > 1 else "state"
    title = sys.argv[2] if len(sys.argv) > 2 else "Listening"
    body = sys.argv[3] if len(sys.argv) > 3 else ""
    timeout_secs = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    bubble = Bubble(title, body, timeout_secs)
    bubble.connect("destroy", Gtk.main_quit)
    bubble.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
