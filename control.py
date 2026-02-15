"""
MIDI Fighter Twister controller helper.

This module provides a MidiFighterTwister class that:
- Connects to a DJ TechTools "MIDI Fighter Twister" using the mido library.
- Listens for Bank 1 knob (encoders 0–15) value changes (MIDI CC values 0–127).
- Exposes convenience methods to read current Bank 1 knob values and colors.
- Provides optional helpers to SET ring LED colors via SysEx (and keeps an
  internal cache so you can also READ them back from the cache). The device
  typically does not report LED colors back to the host, so reading colors is
  best‑effort unless your workflow sets them through this class.

Notes on MIDI mapping (default factory mapping):
- The Twister has 4 banks of 16 encoders (64 total). By default each bank sends
  Control Change (CC) messages with the same CC numbers (0–15) but on different
  MIDI channels:
  • Bank 1 → Channel 0
  • Bank 2 → Channel 1
  • Bank 3 → Channel 2
  • Bank 4 → Channel 3
- This class focuses on Bank 1 (channel 0), CC numbers 0–15.

Colors and SysEx:
- Setting ring LED colors is supported via device-specific SysEx.
- Reading (querying) current LED colors from the device is not officially
  supported in a way that the device reports all colors back on demand.
  Therefore, this class maintains an internal cache of colors you set through
  it; get_bank1_colors() returns those cached values. If you did not set any
  colors, the returned colors may be None.

Dependencies:
- mido (https://mido.readthedocs.io/)
- python-rtmidi (recommended backend for real devices)

Usage example:

    from control import MidiFighterTwister

    with MidiFighterTwister() as twister:
        print("Waiting for knob turns on Bank 1... Press Ctrl+C to exit.")
        twister.on_change(lambda idx, val: print(f"Knob {idx}: {val}"))
        # Optionally set a color
        twister.set_bank1_knob_color(0, (127, 0, 0))  # Red on knob 0
        while True:
            pass  # your loop; or integrate with your app

"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

try:
    import mido
except Exception as e:  # pragma: no cover - environment dependent
    mido = None  # type: ignore
    _mido_import_error = e
else:
    _mido_import_error = None


RGB = Tuple[int, int, int]


class MidiFighterTwister:
    """High-level interface for Bank 1 of a DJTT MIDI Fighter Twister.

    - Auto-discovers the device's input and output ports by name.
    - Listens for CC 0–15 on MIDI channel 0 and tracks their values (0–127).
    - Provides helpers to set ring LED colors via SysEx and to read the cached
      colors.

    Threading model:
    - A background thread reads incoming MIDI messages from the input port and
      updates an internal state dict.

    Context management:
    - Use as a context manager to ensure ports are closed cleanly.
    """

    DEFAULT_DEVICE_SUBSTRINGS = (
        "midi fighter twister",
        "midi fighter twister input",
        "midi fighter twister output",
    )

    def __init__(
        self,
        *,
        bank: int = 1,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        start_reader: bool = True,
    ) -> None:
        if _mido_import_error is not None:  # pragma: no cover
            raise RuntimeError(
                "mido is required but failed to import: {}".format(_mido_import_error)
            )

        if bank != 1:
            # The current requirement is Bank 1, but keep parameter to future-proof
            raise ValueError("This helper currently supports only bank=1")

        self._input_name = input_name
        self._output_name = output_name
        self._inport: Optional[mido.ports.BaseInput] = None
        self._outport: Optional[mido.ports.BaseOutput] = None

        # Bank 1 state: CC index 0–15
        self._values: List[int] = [0] * 16
        self._colors: List[Optional[RGB]] = [None] * 16  # cached, if set via this class
        self._lock = threading.Lock()

        self._callbacks: List[Callable[[int, int], None]] = []

        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Open ports
        self._open_ports()

        if start_reader:
            self._start_reader_thread()

    # ------------- Public API -------------
    def close(self) -> None:
        """Stop background reader and close MIDI ports."""
        self._stop_reader_thread()
        if self._inport is not None:
            try:
                self._inport.close()
            except Exception:
                pass
            self._inport = None
        if self._outport is not None:
            try:
                self._outport.close()
            except Exception:
                pass
            self._outport = None

        # small delay to allow backend to release handles
        time.sleep(0.01)

    def __enter__(self) -> "MidiFighterTwister":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def on_change(self, callback: Callable[[int, int], None]) -> None:
        """Register a callback for knob value changes on Bank 1.

        The callback receives (index, value) where index is 0–15, value is 0–127.
        """
        with self._lock:
            self._callbacks.append(callback)

    def get_bank1_values(self) -> List[int]:
        """Return a copy of the 16 current Bank 1 knob values (0–127)."""
        with self._lock:
            return list(self._values)

    def get_bank1_colors(self) -> List[Optional[RGB]]:
        """Return cached Bank 1 ring LED colors for knobs 0–15.

        Note: The MIDI Fighter Twister usually does not report its LED colors to
        the host. These values reflect what you've set via this class in the
        current Python process. If you haven't set any, items will be None.
        """
        with self._lock:
            return list(self._colors)

    def set_bank1_knob_color(self, index: int, rgb: RGB) -> None:
        """Set ring LED color for a Bank 1 knob (index 0–15) via SysEx.

        rgb components must be integers 0–127.
        """
        self._validate_index(index)
        r, g, b = self._clamp_rgb(rgb)
        self._send_set_color_sysex(index, r, g, b)
        with self._lock:
            self._colors[index] = (r, g, b)

    def set_bank1_colors(self, colors: Iterable[Optional[RGB]]) -> None:
        """Set multiple Bank 1 knob colors from an iterable of length 16.

        Use None to skip a knob.
        """
        for idx, c in enumerate(colors):
            if idx >= 16:
                break
            if c is None:
                continue
            self.set_bank1_knob_color(idx, c)

    # ------------- Internal: MIDI handling -------------
    def _open_ports(self) -> None:
        in_name = self._input_name or self._auto_pick_port(mido.get_input_names())
        out_name = self._output_name or self._auto_pick_port(mido.get_output_names())
        if in_name is None:
            raise RuntimeError(
                "Could not find MIDI Fighter Twister input port. Available: {}".format(
                    mido.get_input_names()
                )
            )
        if out_name is None:
            raise RuntimeError(
                "Could not find MIDI Fighter Twister output port. Available: {}".format(
                    mido.get_output_names()
                )
            )
        self._inport = mido.open_input(in_name)
        self._outport = mido.open_output(out_name)

    def _auto_pick_port(self, names: List[str]) -> Optional[str]:
        lowered = [n for n in names]
        for n in lowered:
            nl = n.lower()
            if any(s in nl for s in self.DEFAULT_DEVICE_SUBSTRINGS):
                return n
        # Fallback: if there's exactly one port, pick it (useful in CI or tests)
        if len(names) == 1:
            return names[0]
        return None

    def _start_reader_thread(self) -> None:
        if self._reader_thread and self._reader_thread.is_alive():
            return
        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="MFT-Reader", daemon=True)
        self._reader_thread.start()

    def _stop_reader_thread(self) -> None:
        if self._reader_thread and self._reader_thread.is_alive():
            self._stop_event.set()
            try:
                # poke the port with an idle wait so thread can exit promptly
                time.sleep(0.01)
            except Exception:
                pass
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
            self._stop_event.clear()

    def _reader_loop(self) -> None:  # pragma: no cover - realtime/hardware dependent
        assert self._inport is not None
        while not self._stop_event.is_set():
            try:
                msg = self._inport.poll()
                if msg is None:
                    time.sleep(0.002)
                    continue
                self._handle_message(msg)
            except Exception:
                # Avoid killing thread due to backend quirks; small backoff
                time.sleep(0.01)

    def _handle_message(self, msg) -> None:
        # Expect ControlChange messages on channel 0, cc 0–15
        if msg.type == "control_change" and getattr(msg, "channel", 0) == 0:
            cc = int(getattr(msg, "control", -1))
            val = int(getattr(msg, "value", 0))
            if 0 <= cc <= 15 and 0 <= val <= 127:
                with self._lock:
                    self._values[cc] = val
                    callbacks = list(self._callbacks)
                # Callbacks outside lock
                for cb in callbacks:
                    try:
                        cb(cc, val)
                    except Exception:
                        pass
        # Ignore other messages silently

    # ------------- Internal: SysEx for colors -------------
    def _send_set_color_sysex(self, index: int, r: int, g: int, b: int) -> None:
        """Send device-specific SysEx to set ring LED color for a knob.

        This uses the widely documented DJ TechTools manufacturer ID 00 01 79 and
        product family bytes associated with the Twister. The exact payload is
        based on community docs; if your device firmware differs, this may need
        adjustment. If sending fails, the method swallows the error to avoid
        breaking value reads.
        """
        if self._outport is None:
            return
        # Construct SysEx: F0 00 01 79 03 04 <enc> <R> <G> <B> F7
        # Where 03 04 represent product/model and the command for setting LED.
        # Encoder index is 0–63 across all banks; for bank 1 we use 0–15.
        manufacturer_id = [0x00, 0x01, 0x79]
        device_and_cmd = [0x03, 0x04]
        data = manufacturer_id + device_and_cmd + [index & 0x7F, r & 0x7F, g & 0x7F, b & 0x7F]
        try:
            sysex = mido.Message('sysex', data=data)
            self._outport.send(sysex)
        except Exception:
            # Some backends/devices may not accept this; ignore to keep app running
            pass

    # ------------- Utilities -------------
    def _validate_index(self, index: int) -> None:
        if not (0 <= index <= 15):
            raise IndexError("Bank 1 knob index must be in 0..15")

    def _clamp_rgb(self, rgb: RGB) -> RGB:
        r, g, b = rgb
        def c(x: int) -> int:
            if x < 0:
                return 0
            if x > 127:
                return 127
            return int(x)
        return c(r), c(g), c(b)


__all__ = ["MidiFighterTwister", "RGB"]
