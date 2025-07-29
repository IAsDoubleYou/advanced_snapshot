import os
import logging
import homeassistant.helpers.config_validation as cv
import voluptuous as vol
import aiofiles
import asyncio
import datetime
import math
import codecs
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.components.camera import async_get_image
from homeassistant.components.camera import async_get_stream_source
from homeassistant.components.ffmpeg import get_ffmpeg_manager
import ffmpeg  
import subprocess
from functools import partial
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from .const import DOMAIN

    
_LOGGER = logging.getLogger(__name__)

# Define the service schema for taking snapshots
SERVICE_SCHEMA = vol.Schema({
    vol.Required("camera_entity_id"): cv.entity_id,
    vol.Required("file_path"): cv.string,
    vol.Optional("file_path_backup"): cv.string,
    # New optional parameters for file rotation
    vol.Optional("base_file_name"): cv.string,  # Basis bestandsnaam voor rotatie (bijv. "camera_snapshot")
    vol.Optional("max_snapshots", default=0): vol.All(vol.Coerce(int), vol.Range(min=0)), # Maximaal aantal snapshots om te bewaren (0 voor geen rotatie)
    vol.Optional("rotate_angle", default=0): vol.All(vol.Coerce(int), vol.Range(min=0, max=360)),
    vol.Optional("crop", default=None): vol.Any(None, [vol.Coerce(int)]),
    vol.Optional("crop_aspect_ratio", default=None): vol.Any(None, vol.Match(r"^\d+:\d+$")),
    vol.Optional("add_bar", default=False): cv.boolean,
    vol.Optional("custom_text_left", default=""): cv.string,
    vol.Optional("custom_text_middle", default=""): cv.string,
    vol.Optional("custom_text_right", default=""): cv.string,
    vol.Optional("setting_font_path", default="/config/custom_components/advanced_snapshot/fonts/Arial.ttf"): cv.string,
    vol.Optional("setting_font_size", default="auto"): vol.Any(vol.Coerce(int), vol.In(["auto"])),
    vol.Optional("setting_font_color", default="black"): cv.string,
    vol.Optional("setting_bar_height", default="40"): vol.Any(vol.Coerce(int), vol.Match(r"^\d+%$")),
    vol.Optional("setting_bar_color", default="white"): cv.string,
    vol.Optional("setting_bar_position", default="bottom"): cv.string
})

# Define the service schema for recording videos (no changes needed here for file rotation)
SERVICE_SCHEMA_RECORD_VIDEO = vol.Schema({
    vol.Required("camera_entity_id"): cv.entity_id,
    vol.Required("file_path"): cv.string,
    vol.Optional("file_path_backup"): cv.string,
    vol.Optional("duration", default=40): vol.All(vol.Coerce(int), vol.Range(min=1, max=40)),
    vol.Optional("rotate_angle", default=0): vol.All(vol.Coerce(int), vol.Range(min=0, max=360)),
    vol.Optional("crop", default=None): vol.Any(None, [vol.Coerce(int)]),
    vol.Optional("crop_aspect_ratio", default=None): vol.Any(None, vol.Match(r"^\d+:\d+$")),
    vol.Optional("add_bar", default=False): cv.boolean,
    vol.Optional("custom_text_left", default=""): cv.string,
    vol.Optional("custom_text_middle", default=""): cv.string,
    vol.Optional("custom_text_right", default=""): cv.string,
    vol.Optional("setting_font_path", default="/config/custom_components/advanced_snapshot/fonts/Arial.ttf"): cv.string,
    vol.Optional("setting_font_size", default="auto"): vol.Any(vol.Coerce(int), vol.In(["auto"])),
    vol.Optional("setting_font_color", default="black"): cv.string,
    vol.Optional("setting_bar_height", default="40"): vol.Any(vol.Coerce(int), vol.Match(r"^\d+%$")),
    vol.Optional("setting_bar_color", default="white"): cv.string,
    vol.Optional("setting_bar_position", default="bottom"): cv.string
})

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the Advanced Snapshot component."""
    _LOGGER.info("Registering the take_snapshot service.")
    hass.services.async_register(
        DOMAIN, "take_snapshot", partial(handle_take_snapshot, hass),
        schema=SERVICE_SCHEMA, supports_response=SupportsResponse.OPTIONAL
    )
    hass.services.async_register(
        DOMAIN, "record_video", partial(handle_record_video, hass),
        schema=SERVICE_SCHEMA_RECORD_VIDEO, supports_response=SupportsResponse.OPTIONAL
    )
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Advanced Snapshot from a config entry."""
    _LOGGER.info(f"Loading Advanced Snapshot configuration: {entry.data}")
    hass.data[DOMAIN] = entry.data
    hass.services.async_register(
        DOMAIN, "take_snapshot", partial(handle_take_snapshot, hass),
        schema=SERVICE_SCHEMA, supports_response=SupportsResponse.OPTIONAL
    )
    hass.services.async_register(
        DOMAIN, "record_video", partial(handle_record_video, hass),
        schema=SERVICE_SCHEMA_RECORD_VIDEO, supports_response=SupportsResponse.OPTIONAL
    )
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading Advanced Snapshot integration.")
    hass.services.async_remove(DOMAIN, "take_snapshot")
    hass.services.async_remove(DOMAIN, "handle_record_video")
    if DOMAIN in hass.data:
        del hass.data[DOMAIN]
    return True

async def handle_take_snapshot(hass: HomeAssistant, call: ServiceCall) -> ServiceResponse:
    """Handle the take_snapshot service call."""
    _LOGGER.info("Received snapshot request.")
    try:
        camera_entity_id = call.data["camera_entity_id"]
        file_path = call.data["file_path"]
        file_path_backup = call.data.get("file_path_backup")
        setting_font_path = call.data.get("setting_font_path")

        # New parameters for file rotation
        base_file_name = call.data.get("base_file_name")
        max_snapshots = call.data.get("max_snapshots", 0) # Default to 0 (no rotation)

        snapshot_folder = hass.data.get(DOMAIN, {}).get("snapshot_folder")
        backup_folder = hass.data.get(DOMAIN, {}).get("backup_folder")
        font_folder = hass.data.get(DOMAIN, {}).get("font_folder")

        # Resolve absolute paths for file_path and font_path
        if not os.path.isabs(file_path):
            file_path = os.path.join(snapshot_folder, file_path)

        if file_path_backup and not os.path.isabs(file_path_backup):
            file_path_backup = os.path.join(backup_folder, file_path_backup)

        if not os.path.isabs(setting_font_path):
            setting_font_path = os.path.join(font_folder, setting_font_path)
        if not os.path.splitext(setting_font_path)[1]:  
            setting_font_path += ".ttf" # Ensure font path has .ttf extension

        rotate_angle = call.data.get("rotate_angle")
        crop = call.data.get("crop")
        crop_aspect_ratio = call.data.get("crop_aspect_ratio")
        add_bar = call.data.get("add_bar", False)
        custom_text_left = call.data.get("custom_text_left", "")
        custom_text_middle = call.data.get("custom_text_middle", "")
        custom_text_right = call.data.get("custom_text_right", "")
        setting_font_size = call.data.get("setting_font_size")
        setting_bar_height = call.data.get("setting_bar_height")
        setting_bar_color = call.data.get("setting_bar_color", "white")
        setting_bar_position = call.data.get("setting_bar_position", "bottom")

        event_data = {
            "success": False,
            "file_path": file_path,
            "backup_path": file_path_backup,
            "original_resolution": None,
            "final_resolution": None,
            "error": None
        }

        # --- Start File Rotation Logic ---
        if base_file_name and max_snapshots > 0:
            _LOGGER.info(f"Applying file rotation for base: '{base_file_name}', max: {max_snapshots}")
            
            # Determine the directory and extension from the original file_path
            target_dir = os.path.dirname(file_path)
            # Use the extension from the original file_path for rotated files
            _, ext = os.path.splitext(file_path) 
            
            # Construct the full path for the base file (the newest snapshot)
            base_full_path = os.path.join(target_dir, f"{base_file_name}{ext}")

            # 1. Delete the oldest snapshot if max_snapshots is reached
            oldest_file_to_delete = os.path.join(target_dir, f"{base_file_name}-{max_snapshots}{ext}")
            if await hass.async_add_executor_job(os.path.exists, oldest_file_to_delete):
                try:
                    await hass.async_add_executor_job(os.remove, oldest_file_to_delete)
                    _LOGGER.debug(f"Deleted oldest snapshot: {oldest_file_to_delete}")
                except OSError as e:
                    _LOGGER.warning(f"Could not delete oldest snapshot {oldest_file_to_delete}: {e}")

            # 2. Shift existing files (e.g., base-1.jpg becomes base-2.jpg, etc.)
            for i in range(max_snapshots - 1, 0, -1):
                old_rotated_path = os.path.join(target_dir, f"{base_file_name}-{i}{ext}")
                new_rotated_path = os.path.join(target_dir, f"{base_file_name}-{i+1}{ext}")
                if await hass.async_add_executor_job(os.path.exists, old_rotated_path):
                    try:
                        await hass.async_add_executor_job(os.rename, old_rotated_path, new_rotated_path)
                        _LOGGER.debug(f"Renamed {old_rotated_path} to {new_rotated_path}")
                    except OSError as e:
                        _LOGGER.warning(f"Could not rename {old_rotated_path} to {new_rotated_path}: {e}")

            # 3. Rename the current 'base_file_name.ext' to 'base_file_name-1.ext'
            # This prepares the slot for the new snapshot
            if await hass.async_add_executor_job(os.path.exists, base_full_path):
                first_rotated_path = os.path.join(target_dir, f"{base_file_name}-1{ext}")
                try:
                    await hass.async_add_executor_job(os.rename, base_full_path, first_rotated_path)
                    _LOGGER.debug(f"Renamed {base_full_path} to {first_rotated_path}")
                except OSError as e:
                    _LOGGER.warning(f"Could not rename {base_full_path} to {first_rotated_path}: {e}")
            
            # Update the file_path to be the base_full_path for the new snapshot
            file_path = base_full_path
            event_data["file_path"] = file_path # Update event_data with the new target path

        # --- End File Rotation Logic ---

        image = await async_get_image(hass, camera_entity_id)
        if image is None or not hasattr(image, "content"):
            _LOGGER.error("Failed to retrieve image from camera.")
            event_data["error"] = "Image could not be retrieved"
            return event_data

        img = Image.open(BytesIO(image.content))
        event_data["original_resolution"] = [img.width, img.height]
        
        if rotate_angle:
            img = img.rotate(rotate_angle, expand=True)
            _LOGGER.info(f"Rotated image by {rotate_angle} degrees")
            
        if crop:
            if len(crop) < 3:
                _LOGGER.error("Invalid crop values: crop must have at least [x, y, width]")
                event_data["error"] = "Invalid crop values"
                return event_data

            x, y, w = crop[:3]
            h = crop[3] if len(crop) == 4 else None

            if crop_aspect_ratio:
                try:
                    aspect_w, aspect_h = map(int, crop_aspect_ratio.split(":"))
                    h = int(w * (aspect_h / aspect_w))
                    _LOGGER.info(f"Using aspect ratio {crop_aspect_ratio}, calculated height: {h}")
                except ValueError:
                    _LOGGER.error(f"Invalid aspect ratio format: {crop_aspect_ratio}")
                    event_data["error"] = "Invalid aspect ratio format"
                    return event_data

            if h is None:
                _LOGGER.error("Height (h) is missing and no aspect ratio provided.")
                event_data["error"] = "Height (h) is missing and no aspect ratio provided."
                return event_data

            if x < 0 or y < 0 or w <= 0 or h <= 0:
                _LOGGER.error(f"Invalid crop dimensions: {x, y, w, h}")
                event_data["error"] = "Invalid crop dimensions"
                return event_data

            if (x + w) > img.width or (y + h) > img.height:
                _LOGGER.error(f"Invalid crop area: ({x}, {y}, {w}, {h}) exceeds image size")
                event_data["error"] = "Invalid crop area"
                return event_data
            
            img = img.crop((x, y, x + w, y + h))

        if add_bar:
            _LOGGER.debug("Adding text bar to image.")
            img = add_text_bar(
                img, custom_text_left, custom_text_middle, custom_text_right,
                setting_font_path, setting_font_size, "black",
                setting_bar_height, setting_bar_color, setting_bar_position, event_data
            )

        event_data["final_resolution"] = [img.width, img.height]

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        await async_save_image(img, file_path)
        _LOGGER.info(f"Snapshot saved at {file_path}")

        if file_path_backup:
            try:
                os.makedirs(os.path.dirname(file_path_backup), exist_ok=True)
                await async_save_image(img, file_path_backup)
                _LOGGER.info(f"Backup snapshot saved at {file_path_backup}")
                event_data["backup_path"] = file_path_backup
            except Exception as e:
                _LOGGER.error(f"Backup failed: {str(e)}")
                event_data["error"] = f"Backup failed: {str(e)}"
                event_data["success"] = False  

        event_data["success"] = True

    except Exception as e:
        _LOGGER.exception(f"Error while taking snapshot: {str(e)}")
        event_data["error"] = str(e)

    return event_data
    
async def handle_record_video(hass: HomeAssistant, call: ServiceCall) -> ServiceResponse:
    """Handle the record_video service call."""
    camera_entity_id = call.data["camera_entity_id"]
    file_path = call.data["file_path"]
    file_path_backup = call.data.get("file_path_backup")
    duration = min(call.data.get("duration", 40), 40)
    rotate_angle = call.data.get("rotate_angle")
    crop = call.data.get("crop")
    crop_aspect_ratio = call.data.get("crop_aspect_ratio")
    add_bar = call.data.get("add_bar", False)
    custom_text_left = call.data.get("custom_text_left", "")
    custom_text_middle = call.data.get("custom_text_middle", "")
    custom_text_right = call.data.get("custom_text_right", "")
    setting_font_path = call.data.get("setting_font_path")
    setting_font_size = call.data.get("setting_font_size", "auto")
    setting_font_color = call.data.get("setting_font_color", "black")
    setting_bar_height = call.data.get("setting_bar_height", "40")
    setting_bar_color = call.data.get("setting_bar_color", "white")
    setting_bar_position = call.data.get("setting_bar_position", "bottom")

    snapshot_folder = hass.data.get(DOMAIN, {}).get("snapshot_folder")
    backup_folder = hass.data.get(DOMAIN, {}).get("backup_folder")
    font_folder = hass.data.get(DOMAIN, {}).get("font_folder")

    if not os.path.isabs(file_path):
        file_path = os.path.join(snapshot_folder, file_path)

    if file_path_backup and not os.path.isabs(file_path_backup):
        file_path_backup = os.path.join(backup_folder, file_path_backup)

    if not os.path.isabs(setting_font_path):
        setting_font_path = os.path.join(font_folder, setting_font_path)
    if not os.path.splitext(setting_font_path)[1]:
        setting_font_path += ".ttf"
    
    stream = await async_get_stream_source(hass, camera_entity_id)
    if not stream:
        _LOGGER.error("Camera stream could not be started.")
        return {"success": False, "error": "Camera stream could not be started"}

    original_resolution = None
    final_resolution = None # Initialize final_resolution
    try:
        probe = await hass.async_add_executor_job(ffmpeg.probe, stream)
        video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        original_resolution = {
            "width": int(video_stream["width"]),
            "height": int(video_stream["height"])
        }
        final_resolution = original_resolution # Start with original resolution for video processing
    except ffmpeg.Error as e:
        _LOGGER.warning(f"Could not probe stream for resolution: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
    except Exception as e:
        _LOGGER.warning(f"Error getting original resolution from stream: {str(e)}")

    stream_input = ffmpeg.input(
        stream, 
        rtsp_transport='tcp',
        timeout='5000000',
        max_delay='500000'
    )
    video = stream_input.video

    if rotate_angle:
        rotation_angle_rad = math.radians(rotate_angle)
        video = video.filter('rotate', rotation_angle_rad, fillcolor='black')
        _LOGGER.info(f"Rotated video by {rotate_angle} degrees")
        # Update final_resolution after rotation if it changes dimensions
        if original_resolution:
            # For 90/270 degrees, width and height swap
            if rotate_angle % 180 != 0:
                final_resolution = {"width": original_resolution["height"], "height": original_resolution["width"]}
            else:
                final_resolution = original_resolution # No dimension change for 0/180

    setting_font_color = sanitize_ffmpeg_color(setting_font_color)
    setting_bar_color = sanitize_ffmpeg_color(setting_bar_color)

    # CROP
    if crop and len(crop) >= 3:
        x, y, w = crop[:3]
        h = crop[3] if len(crop) == 4 else None
        if crop_aspect_ratio and not h:
            try:
                aspect_w, aspect_h = map(int, crop_aspect_ratio.split(":"))
                h = int(w * aspect_h / aspect_w)
                _LOGGER.info(f"Using aspect ratio {crop_aspect_ratio} for video crop, calculated height: {h}")
            except ValueError:
                _LOGGER.error(f"Invalid aspect ratio format for video crop: {crop_aspect_ratio}")
                return {"success": False, "error": "Invalid aspect ratio format"}
        
        if h is None:
            _LOGGER.error("Height (h) is missing for video crop and no aspect ratio provided.")
            return {"success": False, "error": "Height (h) is missing and no aspect ratio provided for video crop."}

        # Validate crop dimensions against current video resolution
        if final_resolution:
            if x < 0 or y < 0 or w <= 0 or h <= 0 or \
               (x + w) > final_resolution["width"] or (y + h) > final_resolution["height"]:
                _LOGGER.error(f"Invalid video crop area: ({x}, {y}, {w}, {h}) exceeds video size {final_resolution['width']}x{final_resolution['height']}")
                return {"success": False, "error": "Invalid video crop dimensions or area"}
        
        video = video.crop(x=x, y=y, width=w, height=h)
        final_resolution = {"width": w, "height": h}
        _LOGGER.info(f"Cropped video to {w}x{h}")

    # BAR + TEXT
    if add_bar and final_resolution:
        now = datetime.datetime.now().strftime("%d.%m.%y %H:%M:%S")

        try:
            if isinstance(setting_bar_height, str) and "%" in setting_bar_height:
                percent = float(setting_bar_height.strip("%")) / 100.0
                bar_height_px = int(final_resolution["height"] * percent)
            else:
                bar_height_px = int(setting_bar_height)
        except ValueError:
            _LOGGER.warning(f"Invalid value for setting_bar_height: {setting_bar_height}. Using default 40px for video.")
            bar_height_px = 40

        bar_y = final_resolution["height"] - bar_height_px if setting_bar_position == "bottom" else 0
        
        # Calculate text_y based on bar_height_px and setting_font_size
        # FFmpeg drawtext filter needs dynamic calculation for vertical centering
        # (h-text_h)/2 refers to height of the bar, not the full video
        text_y_expr = f"({bar_height_px}-text_h)/2 + {bar_y}" 

        if setting_font_size == "auto":
            setting_font_size = max(10, int(bar_height_px * 0.5))
        else:
            setting_font_size = int(setting_font_size)
        
        # Encode text for FFmpeg drawtext filter
        custom_text_left_encoded = utf8_drawtext(custom_text_left)
        custom_text_middle_encoded = utf8_drawtext(custom_text_middle)
        custom_text_right_encoded = utf8_drawtext(custom_text_right)
       
        # Draw the background bar
        video = video.drawbox(
            x=0,
            y=bar_y,
            width="iw",
            height=bar_height_px,
            color=f"{setting_bar_color}",
            t="fill"
        )
        _LOGGER.debug(f"Added bar to video at y={bar_y} with height={bar_height_px}")

        # Add left-aligned text
        if custom_text_left_encoded:
            video = video.drawtext(
                text=custom_text_left_encoded,
                x=10,
                y=text_y_expr,
                fontsize=setting_font_size,
                fontcolor=setting_font_color,
                fontfile=setting_font_path
            )
            _LOGGER.debug(f"Added left text: '{custom_text_left}'")

        # Add middle-aligned text
        if custom_text_middle_encoded:
            video = video.drawtext(
                text=custom_text_middle_encoded,
                x="(w-text_w)/2",
                y=text_y_expr,
                fontsize=setting_font_size,
                fontcolor=setting_font_color,
                fontfile=setting_font_path
            )
            _LOGGER.debug(f"Added middle text: '{custom_text_middle}'")

        # Add right-aligned text
        if custom_text_right_encoded:
            video = video.drawtext(
                text=custom_text_right_encoded,
                x="w-text_w-10",
                y=text_y_expr,
                fontsize=setting_font_size,
                fontcolor=setting_font_color,
                fontfile=setting_font_path
            )
            _LOGGER.debug(f"Added right text: '{custom_text_right}'")

    output_stream = ffmpeg.output(
        video,
        file_path,
        t=duration,
        vcodec="libx264",
        acodec="aac",
        crf=18,
        preset="medium",
        tune="film",          
        pix_fmt="yuv420p",
        format="mp4"
    )
    _LOGGER.info(f"Starting video recording to {file_path} for {duration} seconds.")

    try:
        # Use hass.async_add_executor_job for blocking FFmpeg call
        # This will run FFmpeg in a separate thread, not blocking the Home Assistant event loop
        process = await hass.async_add_executor_job(
            lambda: ffmpeg.run(output_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        out, err = process
        _LOGGER.debug(f"FFmpeg stdout: {out.decode('utf-8')}")
        _LOGGER.debug(f"FFmpeg stderr: {err.decode('utf-8')}")
        _LOGGER.info(f"Video saved at {file_path}")

    except ffmpeg.Error as e:
        _LOGGER.error(f"FFmpeg error during video recording: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
        return {
            "success": False,
            "error": f"FFmpeg error: {e.stderr.decode('utf-8') if e.stderr else str(e)}"
        }
    except Exception as e:
        _LOGGER.exception(f"Unexpected error during video recording: {str(e)}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

    if file_path_backup:
        try:
            os.makedirs(os.path.dirname(file_path_backup), exist_ok=True)
            # Use async_add_executor_job for file copy operation
            await hass.async_add_executor_job(lambda: os.system(f"cp '{file_path}' '{file_path_backup}'"))
            _LOGGER.info(f"Backup video saved at {file_path_backup}")
        except Exception as e:
            _LOGGER.error(f"Backup failed: {str(e)}")
            return {
                "success": False,
                "error": f"Backup failed: {str(e)}"
            }

    return {
        "success": True,
        "file_path": file_path,
        "backup_path": file_path_backup,
        "original_resolution": original_resolution,
        "final_resolution": final_resolution
    }

def sanitize_ffmpeg_color(color_str):
    """Sanitize color string for FFmpeg."""
    color_str = color_str.strip().lower()
    if color_str.startswith("rgb(") and color_str.endswith(")"):
        try:
            r, g, b = map(int, color_str[4:-1].split(","))
            return f"0x{r:02X}{g:02X}{b:02X}"
        except ValueError:
            _LOGGER.warning(f"Invalid RGB color format: {color_str}. Returning original string.")
            return color_str
    return color_str  

    
def utf8_drawtext(text: str) -> str:
    """Encode text for FFmpeg drawtext filter to handle special characters."""
    special_chars = {
        "°": "\\u00B0", # Celsius degree symbol
        "ä": "\\u00E4",
        "ö": "\\u00F6",
        "ü": "\\u00FC",
        "Ä": "\\u00C4",
        "Ö": "\\u00D6",
        "Ü": "\\u00DC",
        "ß": "\\u00DF",
        "€": "\\u20AC", # Euro symbol
        "&": "\\u0026", # Ampersand
        "<": "\\u003C", # Less than
        ">": "\\u003E", # Greater than
        "'": "\\u0027", # Single quote
        "\"": "\\u0022", # Double quote
        "\\": "\\u005C", # Backslash
        ":": "\\u003A", # Colon
        ",": "\\u002C", # Comma
        ";": "\\u003B", # Semicolon
        "=": "\\u003D", # Equals
        "-": "\\u002D", # Hyphen
        "_": "\\u005F", # Underscore
        ".": "\\u002E", # Period
        "/": "\\u002F", # Slash
        "!": "\\u0021", # Exclamation mark
        "?": "\\u003F", # Question mark
        "(": "\\u0028", # Opening parenthesis
        ")": "\\u0029", # Closing parenthesis
        "[": "\\u005B", # Opening bracket
        "]": "\\u005D", # Closing bracket
        "{": "\\u007B", # Opening brace
        "}": "\\u007D", # Closing brace
        "@": "\\u0040", # At symbol
        "#": "\\u0023", # Hash symbol
        "$": "\\u0024", # Dollar symbol
        "%": "\\u0025", # Percent symbol
        "^": "\\u005E", # Caret
        "*": "\\u002A", # Asterisk
        "+": "\\u002B", # Plus
        "~": "\\u007E", # Tilde
        "`": "\\u0060", # Backtick
        "|": "\\u007C"  # Pipe
    }

    # Escape existing backslashes first to prevent double escaping issues
    text = text.replace("\\", "\\\\") 
    for char, unicode_escape in special_chars.items():
        text = text.replace(char, unicode_escape)

    return text # FFmpeg expects the raw unicode escape sequence, not decoded string
    
def add_text_bar_old(img: Image.Image, custom_text_left: str, custom_text_middle: str,
                 custom_text_right: str, setting_font_path: str, setting_font_size,
                 setting_font_color: str, setting_bar_height,
                 setting_bar_color: str, setting_bar_position: str, event_data: dict) -> Image.Image:
    """Old function to add a text bar by extending the image canvas."""
    width, height = img.size

    if isinstance(setting_bar_height, str) and setting_bar_height.endswith('%'):
        try:
            percentage = float(setting_bar_height.strip('%')) / 100.0
            bar_height = int(height * percentage)
        except ValueError:
            _LOGGER.warning(f"Invalid percentage value for setting_bar_height: {setting_bar_height}. Using default 40px.")
            event_data["error"] = f"Invalid percentage value: {setting_bar_height}. Using default 40px."
            bar_height = 40
    else:
        try:
            bar_height = int(setting_bar_height)
        except ValueError:
            _LOGGER.warning(f"Invalid pixel value for setting_bar_height: {setting_bar_height}. Using default 40px.")
            event_data["error"] = f"Invalid pixel value: {setting_bar_height}. Using default 40px."
            bar_height = 40 

    if setting_font_size == "auto":
        setting_font_size = max(10, int(bar_height * 0.5))

    if setting_bar_position == "top":
        new_img = Image.new("RGB", (width, height + bar_height), setting_bar_color)
        new_img.paste(img, (0, bar_height))
        text_y = (bar_height - setting_font_size) // 2
    else:
        new_img = Image.new("RGB", (width, height + bar_height), setting_bar_color)
        new_img.paste(img, (0, 0))
        text_y = height + (bar_height - setting_font_size) // 2

    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype(setting_font_path, setting_font_size)
    except IOError:
        _LOGGER.warning(f"Font file not found: {setting_font_path}, using default font.")
        event_data["error"] = f"Font file not found: {setting_font_path}, using default font."
        font = ImageFont.load_default()

    draw.text((10, text_y), custom_text_left, fill=setting_font_color, font=font)
    draw.text(((width - draw.textlength(custom_text_middle, font=font)) // 2, text_y),
              custom_text_middle, fill=setting_font_color, font=font)
    draw.text((width - draw.textlength(custom_text_right, font=font) - 10, text_y),
              custom_text_right, fill=setting_font_color, font=font)

    return new_img

def add_text_bar(img: Image.Image, custom_text_left: str, custom_text_middle: str,
                 custom_text_right: str, setting_font_path: str, setting_font_size,
                 setting_font_color: str, setting_bar_height,
                 setting_bar_color: str, setting_bar_position: str, event_data: dict) -> Image.Image:
    """Add a text bar directly onto the image without resizing the canvas."""
    width, height = img.size

    if isinstance(setting_bar_height, str) and setting_bar_height.endswith('%'):
        try:
            percentage = float(setting_bar_height.strip('%')) / 100.0
            bar_height = int(height * percentage)
        except ValueError:
            _LOGGER.warning(f"Invalid percentage value for setting_bar_height: {setting_bar_height}. Using default 40px.")
            event_data["error"] = f"Invalid percentage value: {setting_bar_height}. Using default 40px."
            bar_height = 40
    else:
        try:
            bar_height = int(setting_bar_height)
        except ValueError:
            _LOGGER.warning(f"Invalid pixel value for setting_bar_height: {setting_bar_height}. Using default 40px.")
            event_data["error"] = f"Invalid pixel value: {setting_bar_height}. Using default 40px."
            bar_height = 40

    if setting_font_size == "auto":
        setting_font_size = max(10, int(bar_height * 0.5))

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(setting_font_path, setting_font_size)
    except IOError:
        _LOGGER.warning(f"Font file not found: {setting_font_path}, using default font.")
        event_data["error"] = f"Font file not found: {setting_font_path}, using default font."
        font = ImageFont.load_default()

    # Draw bar background
    if setting_bar_position == "top":
        bar_rect = (0, 0, width, bar_height)
        text_y = (bar_height - setting_font_size) // 2
    else:
        bar_rect = (0, height - bar_height, width, height)
        text_y = height - bar_height + (bar_height - setting_font_size) // 2

    draw.rectangle(bar_rect, fill=setting_bar_color)

    margin = 10

    if custom_text_left:
        draw.text((margin, text_y), custom_text_left, fill=setting_font_color, font=font)

    if custom_text_middle:
        text_middle_width = draw.textlength(custom_text_middle, font=font)
        draw.text(((width - text_middle_width) // 2, text_y),
                  custom_text_middle, fill=setting_font_color, font=font)

    if custom_text_right:
        text_right_width = draw.textlength(custom_text_right, font=font)
        draw.text((width - text_right_width - margin, text_y),
                  custom_text_right, fill=setting_font_color, font=font)

    return img

async def async_save_image(img: Image.Image, file_path: str):
    """Save the PIL image to the specified file path asynchronously."""
    ext = os.path.splitext(file_path)[1].lower()
    format_map = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
    image_format = format_map.get(ext, "JPEG")

    async with aiofiles.open(file_path, "wb") as f:
        buffer = BytesIO()
        img.save(buffer, format=image_format)
        buffer.seek(0)
        await f.write(buffer.getvalue())

    _LOGGER.info(f"Snapshot saved: {file_path} ({image_format})")
