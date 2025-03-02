# main.py - DEESEEK GB Emulator System with PyBoy
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
import base64
import json
import os
import time
import threading
import numpy as np  # Make sure numpy is imported
from PIL import Image
import io
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Setup directories
ROMS_DIR = os.path.join(os.getcwd(), 'roms')
SAVES_DIR = os.path.join(os.getcwd(), 'saves')
SCREENSHOTS_DIR = os.path.join(os.getcwd(), 'screenshots')

# Ensure directories exist
for directory in [ROMS_DIR, SAVES_DIR, SCREENSHOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Global variables to hold emulator state
emulator = None
emulator_thread = None
current_rom_path = None
emulation_running = False
frame_count = 0

class EmulatorInstance:
    """Wrapper for the PyBoy emulator"""

    def __init__(self, rom_path=None):
        self.rom_path = rom_path
        self.pyboy = None
        self.frame_buffer = None
        self.screen_size = (160, 144)  # Default GB resolution
        self.audio_enabled = False
        self.audio_buffer = []
        self.button_map = {
            'up': "up",
            'down': "down",
            'left': "left",
            'right': "right",
            'a': "a",
            'b': "b",
            'start': "start",
            'select': "select"
        }

    def initialize(self):
        """Initialize PyBoy emulator with the provided ROM"""
        if not self.rom_path or not os.path.exists(self.rom_path):
            logger.error(f"ROM file not found: {self.rom_path}")
            return False

        try:
            # Import PyBoy here to prevent issues if the package is not installed
            from pyboy import PyBoy

            logger.info(f"Loading ROM: {self.rom_path}")

            # Make sure the ROM file is readable
            with open(self.rom_path, 'rb') as f:
                rom_data = f.read(256)  # Read first 256 bytes to check file is accessible
                logger.info(f"ROM file is readable, size: {os.path.getsize(self.rom_path)} bytes")

            # Try different initialization options
            logger.info("Attempting to initialize PyBoy with alternate settings...")

            # Set environment variables to help with display issues
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"

            # First try with newer PyBoy syntax (window="null") with sound emulation
            try:
                self.pyboy = PyBoy(
                    self.rom_path,
                    window="null",
                    sound=False,  # No audio output device
                    sound_emulated=True,  # But emulate sound for capture
                    debug=True,
                )
                logger.info("Successfully initialized PyBoy with window=null and sound emulation")
                self.audio_enabled = True
            except Exception as null_err:
                logger.warning(f"Null window mode failed: {null_err}, trying window_type=dummy...")

                try:
                    # Try with older PyBoy syntax (window_type="dummy")
                    self.pyboy = PyBoy(
                        self.rom_path,
                        window_type="dummy",
                        sound=False,
                        sound_emulated=True,  # Enable sound emulation
                        debug=True,
                    )
                    logger.info("Successfully initialized PyBoy with window_type=dummy and sound emulation")
                    self.audio_enabled = True
                except Exception as dummy_err:
                    logger.warning(f"Dummy window mode failed: {dummy_err}, trying headless mode...")

                    # Last attempt with headless mode
                    self.pyboy = PyBoy(
                        self.rom_path,
                        window_type="headless",
                        sound=False,
                        sound_emulated=True,  # Enable sound emulation
                        debug=True,
                    )
                    logger.info("Successfully initialized PyBoy with window_type=headless and sound emulation")
                    self.audio_enabled = True

            # Try to set emulation speed
            try:
                self.pyboy.set_emulation_speed(1)  # Normal speed
            except Exception as speed_err:
                logger.warning(f"Could not set emulation speed: {speed_err}")

            # Check if emulator is initialized correctly
            if not hasattr(self.pyboy, 'tick'):
                raise Exception("PyBoy instance doesn't have the expected 'tick' method")

            # Start emulator - just one tick to see if it works
            self.pyboy.tick()
            logger.info(f"Emulator started successfully with ROM: {self.rom_path}")
            return True

        except ImportError as e:
            logger.error(f"PyBoy import error - package not installed correctly: {e}")
            # Try to show what's available
            try:
                import sys, pkgutil
                logger.info(f"Available modules: {[name for _, name, _ in pkgutil.iter_modules()]}")
            except:
                pass
            return False
        except PermissionError as e:
            logger.error(f"Permission denied when accessing ROM file: {e}")
            return False
        except Exception as e:
            import traceback
            logger.error(f"Failed to initialize emulator: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Check for SDL-related errors
            if "SDL" in str(e) or "sdl" in str(e).lower():
                logger.error("SDL2 initialization error detected. Checking SDL configuration...")
                # Try to load SDL to see if it works
                try:
                    import sdl2
                    logger.info(f"SDL2 is available")
                    logger.info(f"SDL2 path: {sdl2.__file__}")

                    # Check SDL environment
                    sdl_env_vars = {k: v for k, v in os.environ.items() if 'SDL' in k}
                    logger.info(f"SDL environment variables: {sdl_env_vars}")

                except Exception as sdl_err:
                    logger.error(f"SDL2 import error: {sdl_err}")

            # Special handling for ROM format errors
            if "ROM" in str(e) and ("format" in str(e).lower() or "invalid" in str(e).lower()):
                logger.error("The ROM file may be corrupted or in an unsupported format")
                # Try to inspect ROM header
                try:
                    with open(self.rom_path, 'rb') as f:
                        header = f.read(80)
                        logger.info(f"ROM header (hex): {header.hex()[:50]}...")
                except:
                    pass

            return False

    def get_screen_buffer(self):
        """Get the current screen buffer as a PIL image"""
        if not self.pyboy:
            return None

        try:
            # Try all known methods to get screen data with improved rendering
            methods_to_try = [
                self._get_screen_image_direct,     # Method 4: Direct screen_image method (prioritized for better sprites)
                self._get_screen_game_area,        # Method 5: Game area method (often better rendering)
                self._get_screen_direct,           # Method 1: Direct .screen attribute access
                self._get_screen_lcd,              # Method 2: LCD screen access
                self._get_screen_botsupport,       # Method 3: BotSupport API
            ]

            # Try each method until one works
            for method in methods_to_try:
                try:
                    result = method()
                    if result is not None:
                        # If we get a numpy array, make sure we correctly convert it to an image
                        if isinstance(result, np.ndarray):
                            # Make a copy to avoid potential memory issues
                            img_array = result.copy()

                            # Check if array needs to be normalized
                            if img_array.dtype != np.uint8:
                                # Scale to 0-255 range for proper display
                                img_array = ((img_array - img_array.min()) * (255.0 / max(1, img_array.max() - img_array.min()))).astype(np.uint8)

                            # Handle different array shapes
                            if len(img_array.shape) == 2:  # Grayscale
                                return Image.fromarray(img_array, mode='L')
                            elif len(img_array.shape) == 3:
                                if img_array.shape[2] == 3:  # RGB
                                    return Image.fromarray(img_array, mode='RGB')
                                elif img_array.shape[2] == 4:  # RGBA
                                    return Image.fromarray(img_array, mode='RGBA')

                            # Fallback for other shapes
                            return Image.fromarray(img_array)

                        # Return PIL Image directly
                        return result
                except Exception as e:
                    logger.debug(f"Screen method failed: {e}")
                    continue

            # Create a fallback screen if all methods fail
            logger.warning("All screen buffer access methods failed, using fallback screen")
            width, height = 160, 144
            return Image.new('RGB', (width, height), color='green')

        except Exception as e:
            logger.error(f"Failed to get screen buffer: {e}")
            # Return a fallback screen
            width, height = 160, 144
            return Image.new('RGB', (width, height), color='green')

    # Individual screen access methods
    def _get_screen_direct(self):
        """Method 1: Direct screen attribute access"""
        if hasattr(self.pyboy, 'screen'):
            from PIL import Image

            screen_buffer = self.pyboy.screen
            if isinstance(screen_buffer, bytes):
                # Try different modes based on length
                width, height = 160, 144
                bytes_per_pixel = len(screen_buffer) // (width * height)

                if bytes_per_pixel == 3:
                    img = Image.frombytes('RGB', (width, height), screen_buffer)
                    # Apply color enhancement for Game Boy sprites
                    return self._enhance_gameboy_colors(img)
                elif bytes_per_pixel == 4:
                    img = Image.frombytes('RGBA', (width, height), screen_buffer)
                    return self._enhance_gameboy_colors(img)
                elif bytes_per_pixel == 1:
                    # For grayscale, convert to proper Game Boy palette
                    img = Image.frombytes('L', (width, height), screen_buffer)
                    return self._apply_gameboy_palette(img)
            elif isinstance(screen_buffer, np.ndarray):
                # Enhanced array processing with proper Game Boy color handling
                try:
                    if len(screen_buffer.shape) == 3 and (screen_buffer.shape[2] == 3 or screen_buffer.shape[2] == 4):
                        # Properly handle RGB/RGBA arrays
                        if screen_buffer.dtype != np.uint8:
                            screen_buffer = (screen_buffer * 255).astype(np.uint8)
                        img = Image.fromarray(screen_buffer)
                        return self._enhance_gameboy_colors(img)
                except (IndexError, AttributeError):
                    # Handle 2D arrays (grayscale)
                    if len(screen_buffer.shape) == 2:
                        if screen_buffer.dtype != np.uint8:
                            screen_buffer = (screen_buffer * 255).astype(np.uint8)
                        img = Image.fromarray(screen_buffer, mode='L')
                        return self._apply_gameboy_palette(img)

                # Last resort: try to convert the array directly
                try:
                    return Image.fromarray(np.asarray(screen_buffer).astype('uint8'))
                except:
                    pass
                return screen_buffer
            elif isinstance(screen_buffer, Image.Image):
                return self._enhance_gameboy_colors(screen_buffer)
        return None

    def _enhance_gameboy_colors(self, img):
        """Enhance colors to make Game Boy sprites look better"""
        if img is None:
            return None

        try:
            # Convert to RGBA if not already
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Apply enhanced color processing for better sprite rendering
            from PIL import ImageEnhance, ImageFilter

            # Color balance adjustment
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.1)  # Slightly enhance color
            
            # Improved contrast for better sprite definition
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)  # Moderately increase contrast
            
            # Apply slight brightness adjustment to prevent dark sprites
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)  # Slightly increase brightness

            # Enhanced sharpness for pixel-perfect rendering
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)  # Higher sharpness for clearer pixels
            
            # Apply subtle unsharp mask for better edge definition
            img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=150, threshold=3))
            
            return img
        except Exception as e:
            logger.debug(f"Color enhancement failed: {e}")
            return img

    def _apply_gameboy_palette(self, img):
        """Apply classic Game Boy color palette to grayscale image"""
        if img is None:
            return None

        try:
            # Improved authentic Game Boy palette (light to dark)
            # Using more accurate DMG (original Game Boy) colors
            gb_colors = [
                (155, 188, 15),  # Lightest (background)
                (139, 172, 15),  # Light
                (48, 98, 48),    # Dark
                (15, 56, 15)     # Darkest
            ]
            
            # Alternative palettes that can be selected programmatically
            gb_pocket_colors = [
                (156, 186, 165),  # Lightest
                (140, 165, 144),  # Light
                (52, 101, 90),    # Dark
                (8, 24, 32)       # Darkest
            ]
            
            gb_bivert_colors = [
                (224, 248, 208),  # Lightest (bivert mod)
                (136, 192, 112),  # Light
                (52, 104, 86),    # Dark
                (8, 24, 32)       # Darkest
            ]
            
            # Use the standard DMG palette by default
            selected_palette = gb_colors
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Create a palette image
            from PIL import Image
            palette_img = Image.new('P', (1, 1))
            palette = []

            # Fill palette with Game Boy colors and pad to 256
            for r, g, b in selected_palette:
                palette.extend((r, g, b))
            palette.extend([0, 0, 0] * (256 - len(selected_palette)))

            palette_img.putpalette(palette)

            # Apply dithering to better represent color transitions
            return img.quantize(palette=palette_img, dither=Image.FLOYDSTEINBERG)
        except Exception as e:
            logger.debug(f"Palette application failed: {e}")
            return img

    def _get_screen_lcd(self):
        """Method 2: LCD attribute access"""
        if hasattr(self.pyboy, 'lcd'):
            lcd = self.pyboy.lcd
            if hasattr(lcd, 'pixels'):
                pixels = lcd.pixels
                if isinstance(pixels, np.ndarray):
                    return pixels
        return None

    def _get_screen_botsupport(self):
        """Method 3: BotSupport API access"""
        if hasattr(self.pyboy, 'botsupport_manager'):
            manager = self.pyboy.botsupport_manager()
            if hasattr(manager, 'screen'):
                screen = manager.screen()
                if hasattr(screen, 'screen_image'):
                    return screen.screen_image()
        return None

    def _get_screen_image_direct(self):
        """Method 4: Direct screen_image method"""
        if hasattr(self.pyboy, 'screen_image'):
            return self.pyboy.screen_image()
        return None

    def _get_screen_game_area(self):
        """Method 5: Game area method"""
        if hasattr(self.pyboy, 'game_area') and callable(self.pyboy.game_area):
            game_area = self.pyboy.game_area()
            return game_area
        return None

    def get_screen_base64(self):
        """Get the current screen as a base64 encoded string"""
        image = self.get_screen_buffer()
        if image is None:
            return None

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            try:
                from PIL import Image
                image = Image.fromarray(image)
            except Exception as e:
                logger.error(f"Failed to convert numpy array to PIL Image: {e}")
                return None

        try:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image to base64: {e}")
            return None

    def get_audio_data(self):
        """Get audio data from the emulator if available"""
        if not self.pyboy or not self.audio_enabled:
            return None

        try:
            # Try different methods to access audio data based on PyBoy version
            audio_data = None

            # Method 1: Direct audio buffer access if available
            if hasattr(self.pyboy, 'get_audio_buffer'):
                audio_data = self.pyboy.get_audio_buffer()

            # Method 2: Access through sound module
            elif hasattr(self.pyboy, 'sound') and hasattr(self.pyboy.sound, 'get_audio_buffer'):
                audio_data = self.pyboy.sound.get_audio_buffer()

            # Method 3: Access through botsupport API
            elif hasattr(self.pyboy, 'botsupport_manager'):
                try:
                    manager = self.pyboy.botsupport_manager()
                    if hasattr(manager, 'sound'):
                        sound = manager.sound()
                        if hasattr(sound, 'get_audio_buffer'):
                            audio_data = sound.get_audio_buffer()
                except Exception as e:
                    logger.debug(f"Botsupport audio access error: {e}")

            if audio_data is not None and len(audio_data) > 0:
                # Convert audio data to base64 for transport
                try:
                    # Normalize audio data if needed
                    if isinstance(audio_data, np.ndarray):
                        # Make sure it's in the right format for Web Audio API
                        if audio_data.dtype != np.int16:
                            audio_data = (audio_data * 32767).astype(np.int16)

                    # Convert to bytes if it's numpy array
                    if isinstance(audio_data, np.ndarray):
                        audio_bytes = audio_data.tobytes()
                    else:
                        audio_bytes = audio_data

                    # Encode to base64
                    return base64.b64encode(audio_bytes).decode('utf-8')
                except Exception as e:
                    logger.error(f"Audio data conversion error: {e}")

            return None

        except Exception as e:
            logger.error(f"Error getting audio data: {e}")
            return None

    def press_button(self, button_name):
        """Press a button on the emulator"""
        if not self.pyboy or button_name.lower() not in self.button_map:
            return False

        try:
            # Try to get button codes from PyBoy in different ways
            # First, try direct constants if available in this version
            button_map_name = self.button_map[button_name.lower()].upper()

            # Method 1: Try using WindowEvent directly
            try:
                # Try multiple ways to import WindowEvent
                try:
                    from pyboy import WindowEvent
                except ImportError:
                    # In newer versions, WindowEvent might be in a different location
                    try:
                        from pyboy.pyboy import WindowEvent
                    except ImportError:
                        # As a last resort, check if it's directly on the PyBoy instance
                        if hasattr(self.pyboy, 'WindowEvent'):
                            WindowEvent = self.pyboy.WindowEvent
                        else:
                            raise ImportError("WindowEvent not found")

                button = getattr(WindowEvent, f"PRESS_{button_map_name}")
                release = getattr(WindowEvent, f"RELEASE_{button_map_name}")

                self.pyboy.send_input(button)
                self.pyboy.tick()
                self.pyboy.send_input(release)
                self.pyboy.tick()
                return True
            except (ImportError, AttributeError) as e1:
                logger.debug(f"Method 1 failed: {e1}")

                # Method 2: Try with constants at the PyBoy level
                try:
                    button = getattr(self.pyboy, f"BUTTON_{button_map_name}")

                    self.pyboy.button_press(button)
                    self.pyboy.tick()
                    self.pyboy.button_release(button)
                    self.pyboy.tick()
                    return True
                except AttributeError as e2:
                    logger.debug(f"Method 2 failed: {e2}")

                    # Method 3: Handle via string names using the botsupport API
                    try:
                        botsupport = self.pyboy.botsupport_manager()
                        if hasattr(botsupport, 'buttons'):
                            button_obj = botsupport.buttons()
                            getattr(button_obj, f"press_{button_map_name.lower()}")()
                            self.pyboy.tick()
                            getattr(button_obj, f"release_{button_map_name.lower()}")()
                            self.pyboy.tick()
                            return True
                    except Exception as e3:
                        logger.debug(f"Method 3 failed: {e3}")

            logger.error(f"All button press methods failed for {button_name}")
            return False

        except Exception as e:
            logger.error(f"Failed to press button {button_name}: {e}")
            return False

    def tick(self):
        """Run a single emulator frame"""
        if not self.pyboy:
            return False

        try:
            self.pyboy.tick()
            return True
        except Exception as e:
            logger.error(f"Emulator tick failed: {e}")
            return False

    def save_state(self, path=None):
        """Save emulator state to a file"""
        if not self.pyboy:
            return False

        try:
            if not path:
                rom_name = os.path.basename(self.rom_path).split('.')[0]
                path = os.path.join(SAVES_DIR, f"{rom_name}_save.state")

            self.pyboy.save_state(path)
            logger.info(f"State saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def load_state(self, path=None):
        """Load emulator state from a file"""
        if not self.pyboy:
            return False

        try:
            if not path:
                rom_name = os.path.basename(self.rom_path).split('.')[0]
                path = os.path.join(SAVES_DIR, f"{rom_name}_save.state")

            if not os.path.exists(path):
                logger.warning(f"No save state found at {path}")
                return False

            self.pyboy.load_state(path)
            logger.info(f"State loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def take_screenshot(self):
        """Save current screen as a screenshot file"""
        image = self.get_screen_buffer()
        if not image:
            return None

        try:
            timestamp = int(time.time())
            rom_name = os.path.basename(self.rom_path).split('.')[0]
            filename = f"{rom_name}_{timestamp}.png"
            path = os.path.join(SCREENSHOTS_DIR, filename)

            image.save(path)
            logger.info(f"Screenshot saved to {path}")
            return path

        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None

    def cleanup(self):
        """Clean up emulator resources"""
        if self.pyboy:
            try:
                self.pyboy.stop()
                logger.info("Emulator stopped")
            except:
                pass
            self.pyboy = None

def emulator_loop():
    """Main loop for the emulator thread"""
    global emulator, emulation_running, frame_count
    
    # Calculate timing for consistent 60fps emulation
    target_frame_time = 1/60  # Target 60fps (Game Boy's native refresh rate)
    last_frame_time = time.time()

    while emulation_running and emulator and emulator.pyboy:
        try:
            # Measure start time for this frame
            frame_start = time.time()
            
            # Run a frame
            emulator.tick()
            frame_count += 1

            # Every frame, send the screen to clients for maximum smoothness
            # Game Boy runs at 59.7fps, so we want to match that for authentic rendering
            try:
                # Get and process screen with enhanced sprite rendering
                screen_data = emulator.get_screen_base64()
                if screen_data is not None and len(screen_data) > 0:
                    socketio.emit('screen_update', {'screen': screen_data})
            except Exception as screen_err:
                logger.error(f"Error sending screen update: {screen_err}")

            # Every 3 frames, send audio data if available (audio needs less frequent updates)
            if frame_count % 3 == 0 and emulator.audio_enabled:
                try:
                    audio_data = emulator.get_audio_data()
                    if audio_data is not None:
                        socketio.emit('audio_data', {'audio': audio_data})
                except Exception as audio_err:
                    logger.error(f"Error sending audio update: {audio_err}")

            # Every 60 frames (about 1 second), perform enhanced VRAM refresh to fix rendering artifacts
            if frame_count % 60 == 0:
                try:
                    # Advanced sprite refresh technique
                    # This helps with sprite flickering issues common in GB emulation
                    for _ in range(3):  # Increased refresh cycles
                        emulator.tick()
                        time.sleep(0.001)  # Small delay between refreshes
                except Exception:
                    pass
            
            # Precise frame timing for smooth 60fps
            frame_end = time.time()
            frame_duration = frame_end - frame_start
            sleep_time = max(0, target_frame_time - frame_duration)
            
            # Sleep just enough to maintain proper frame rate
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in emulator loop: {e}")
            time.sleep(0.1)

    logger.info("Emulator loop stopped")

# ------------------- Flask Routes -------------------

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/roms', methods=['GET'])
def list_roms():
    """List available ROMs"""
    roms = []
    for file in os.listdir(ROMS_DIR):
        if file.lower().endswith(('.gb', '.gbc')):
            roms.append({
                'filename': file,
                'path': os.path.join(ROMS_DIR, file)
            })
    return jsonify({'roms': roms})

@app.route('/api/load_rom', methods=['POST'])
def load_rom():
    """Load a ROM file into the emulator"""
    global emulator, emulator_thread, current_rom_path, emulation_running

    # Handle ROM file upload
    if 'rom_file' in request.files:
        rom_file = request.files['rom_file']
        if rom_file.filename == '':
            return jsonify({"error": "No ROM file selected"}), 400

        # Save the ROM file
        rom_path = os.path.join(ROMS_DIR, rom_file.filename)
        try:
            rom_file.save(rom_path)
            logger.info(f"ROM file saved to {rom_path}")

            # Validate ROM file
            try:
                with open(rom_path, 'rb') as f:
                    header = f.read(80)
                    # Check file size
                    file_size = os.path.getsize(rom_path)
                    logger.info(f"ROM file size: {file_size} bytes")

                    # Simple validation - check for Nintendo logo in the header (mandatory in GB ROMs)
                    # The Nintendo logo is at offset 0x104-0x133 in GB ROM headers
                    f.seek(0x104)
                    nintendo_logo = f.read(48)
                    # First few bytes of the Nintendo logo are usually consistent
                    if len(nintendo_logo) >= 4 and nintendo_logo[0:4] != b'\xCE\xED\x66\x66':
                        logger.warning("ROM may not have a valid Nintendo logo in header")
            except Exception as validate_err:
                logger.warning(f"ROM validation warning (non-fatal): {validate_err}")

        except Exception as e:
            logger.error(f"Failed to save ROM file: {e}")
            return jsonify({"error": f"Failed to save ROM file: {str(e)}"}), 500
    else:
        # Load existing ROM by filename
        data = request.json or {}
        filename = data.get('filename')
        if not filename:
            return jsonify({"error": "No ROM filename provided"}), 400

        rom_path = os.path.join(ROMS_DIR, filename)
        if not os.path.exists(rom_path):
            return jsonify({"error": f"ROM file not found: {filename}"}), 404

        # Validate the ROM file quickly
        try:
            logger.info(f"Validating ROM file: {rom_path}")
            file_size = os.path.getsize(rom_path)
            logger.info(f"ROM file size: {file_size} bytes")
            if file_size < 10000:
                logger.warning(f"ROM file may be too small ({file_size} bytes)")
        except Exception as e:
            logger.warning(f"ROM validation warning: {e}")

    # Stop existing emulator if running
    stop_emulator()

    # Initialize new emulator
    current_rom_path = rom_path
    emulator = EmulatorInstance(rom_path)

    # Start emulator
    logger.info(f"Initializing emulator with ROM: {rom_path}")
    if emulator.initialize():
        # Start emulator thread
        emulation_running = True
        emulator_thread = threading.Thread(target=emulator_loop)
        emulator_thread.daemon = True
        emulator_thread.start()
        logger.info(f"Emulator thread started for ROM: {os.path.basename(rom_path)}")

        return jsonify({
            "status": "ROM loaded",
            "filename": os.path.basename(rom_path)
        })
    else:
        logger.error(f"Failed to initialize emulator with ROM: {rom_path}")
        # Check if PyBoy is properly installed
        try:
            import pyboy
            # Different versions of PyBoy have different version attributes
            if hasattr(pyboy, '__version__'):
                logger.info(f"PyBoy version: {pyboy.__version__}")
            elif hasattr(pyboy, 'version'):
                logger.info(f"PyBoy version: {pyboy.version}")
            else:
                logger.info("PyBoy found, but version unknown")
        except ImportError:
            logger.error("PyBoy is not properly installed")
            return jsonify({"error": "PyBoy emulator module is not properly installed"}), 500

        return jsonify({"error": "Failed to initialize emulator. Check server logs for details."}), 500

@app.route('/api/press_button', methods=['POST'])
def press_button():
    """Press a button on the emulator"""
    global emulator

    if not emulator or not emulator.pyboy:
        return jsonify({"error": "Emulator not running"}), 400

    data = request.json
    button = data.get('button')
    if not button:
        return jsonify({"error": "No button specified"}), 400

    success = emulator.press_button(button)
    return jsonify({"success": success})

@app.route('/api/screenshot', methods=['POST'])
def take_screenshot():
    """Take a screenshot of the current emulator screen"""
    global emulator

    if not emulator or not emulator.pyboy:
        return jsonify({"error": "Emulator not running"}), 400

    screenshot_path = emulator.take_screenshot()
    if screenshot_path:
        return jsonify({
            "status": "Screenshot taken",
            "path": screenshot_path,
            "filename": os.path.basename(screenshot_path)
        })
    else:
        return jsonify({"error": "Failed to take screenshot"}), 500

@app.route('/api/save_state', methods=['POST'])
def save_state():
    """Save the current emulator state"""
    global emulator

    if not emulator or not emulator.pyboy:
        return jsonify({"error": "Emulator not running"}), 400

    success = emulator.save_state()
    return jsonify({"status": "State saved" if success else "Failed to save state"})

@app.route('/api/load_state', methods=['POST'])
def load_state():
    """Load a saved emulator state"""
    global emulator

    if not emulator or not emulator.pyboy:
        return jsonify({"error": "Emulator not running"}), 400

    success = emulator.load_state()
    return jsonify({"status": "State loaded" if success else "No save state found"})

@app.route('/api/get_screen', methods=['GET'])
def get_screen():
    """Get the current emulator screen"""
    global emulator

    if not emulator or not emulator.pyboy:
        return jsonify({"error": "Emulator not running"}), 400

    screen_data = emulator.get_screen_base64()
    if screen_data:
        return jsonify({"screen": screen_data})
    else:
        return jsonify({"error": "Failed to get screen"}), 500

@app.route('/api/get_audio', methods=['GET'])
def get_audio():
    """Get the current emulator audio data"""
    global emulator

    if not emulator or not emulator.pyboy:
        return jsonify({"error": "Emulator not running"}), 400

    if not emulator.audio_enabled:
        return jsonify({"error": "Audio not enabled for this emulator instance"}), 400

    audio_data = emulator.get_audio_data()
    if audio_data:
        return jsonify({"audio": audio_data})
    else:
        return jsonify({"error": "Failed to get audio data"}), 500

@app.route('/api/debug_screen', methods=['GET'])
def debug_screen():
    """Debug the screen rendering"""
    global emulator

    debug_info = {
        "emulator_running": emulator is not None and emulator.pyboy is not None,
        "frame_count": frame_count,
        "pyboy_attrs": []
    }

    if emulator and emulator.pyboy:
        # Collect PyBoy attributes
        try:
            debug_info["pyboy_attrs"] = dir(emulator.pyboy)

            # Check for screen-related attributes
            screen_attrs = {}
            if hasattr(emulator.pyboy, 'screen'):
                screen = emulator.pyboy.screen
                screen_attrs["screen_type"] = str(type(screen))
                if isinstance(screen, np.ndarray):
                    screen_attrs["screen_shape"] = screen.shape
                    screen_attrs["screen_dtype"] = str(screen.dtype)
                elif isinstance(screen, bytes):
                    screen_attrs["screen_bytes_len"] = len(screen)

            debug_info["screen_attrs"] = screen_attrs

            # Try to generate a test image
            test_img = emulator.get_screen_buffer()
            if test_img:
                debug_info["test_image_size"] = test_img.size
                debug_info["test_image_mode"] = test_img.mode
        except Exception as e:
            debug_info["error"] = str(e)

    return jsonify(debug_info)

@app.route('/api/screenshots/<filename>', methods=['GET'])
def get_screenshot(filename):
    """Serve a screenshot file"""
    return send_file(os.path.join(SCREENSHOTS_DIR, filename))

def stop_emulator():
    """Stop the emulator and clean up resources"""
    global emulator, emulator_thread, emulation_running

    emulation_running = False
    if emulator_thread and emulator_thread.is_alive():
        emulator_thread.join(timeout=2.0)

    if emulator:
        emulator.cleanup()
        emulator = None

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

@socketio.on('get_screen')
def handle_get_screen():
    """Send the current screen to the client"""
    global emulator

    if emulator and emulator.pyboy:
        screen_data = emulator.get_screen_base64()
        if screen_data:
            return {'screen': screen_data}
    return {'error': 'Emulator not running'}

# Clean up when the server exits
import atexit
atexit.register(stop_emulator)

if __name__ == '__main__':
    # Ensure the template directory exists
    os.makedirs('templates', exist_ok=True)

    # Copy index.html to templates directory if it exists and is newer
    if os.path.exists('index.html'):
        root_mtime = os.path.getmtime('index.html')
        template_path = 'templates/index.html'
        if not os.path.exists(template_path) or os.path.getmtime(template_path) < root_mtime:
            logger.info("Copying updated index.html to templates directory")
            with open(template_path, 'w') as f:
                with open('index.html', 'r') as source:
                    f.write(source.read())

    # Start the server
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
