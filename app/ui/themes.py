# ==============================================================================
# Program: themes.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Application theme engine and CSS/QSS stylesheet provider.
#          Applies Heavenscape dark palette tokens, border radiuses, and high-
#          contrast accessibility themes.
# ==============================================================================

"""
Karl design system — customizable theme engine.
Palettes are compiled dynamically into QSS.
"""

# Fallback accent colour (neon cyan) used when no custom accent is chosen.
ACCENT_DEFAULT = "#00E5FF"

# Theme registry: name -> raw palette. Every entry supplies the same keys:
#   accent / accent_alt   highlight colour and its darker partner
#   bg_deep/surface/raised  three background elevations (back to front)
#   border / border_hi    normal and emphasised outlines
#   text_hi/mid/lo        primary, secondary and muted text
#   success/warning/danger  status colours
#   glow_strength         multiplier for neon glow effects
#   motion_style          animation flavour ("normal" or "pulse")
# get_theme_colors() derives the final palette from these, so adding a theme
# only requires adding one entry here.
THEMES = {
    # Default theme: deep navy/black with neon cyan outlines. Also the fallback when a name is unknown.
    "Karl Obsidian Core": {
        "description": "Default cybernetic dark navy and black interface with sharp neon cyan outlines.",
        "accent": "#00E5FF",
        "accent_alt": "#0099AA",
        "bg_deep": "#020205",
        "bg_surface": "#0D0D1B",
        "bg_raised": "#14142D",
        "border": "#1F1F3D",
        "border_hi": "#35356E",
        "text_hi": "#F0F5FF",
        "text_mid": "#A0AEC0",
        "text_lo": "#6A7B95",
        "success": "#00FFAA",
        "warning": "#FFCC00",
        "danger": "#FF3366",
        "glow_strength": 1.0,
        "motion_style": "normal"
    },
    # Deep ocean blues with cyan accents and cold white text; mild pulse motion.
    "Abyssal Blue Engine": {
        "description": "Deep ocean blue base, high-energy cyan accents, and cold white text layers.",
        "accent": "#00FFFF",
        "accent_alt": "#008899",
        "bg_deep": "#000814",
        "bg_surface": "#001D3D",
        "bg_raised": "#003566",
        "border": "#004B87",
        "border_hi": "#0066CC",
        "text_hi": "#FFFFFF",
        "text_mid": "#8ECAE6",
        "text_lo": "#219EBC",
        "success": "#2A9D8F",
        "warning": "#E9C46A",
        "danger": "#E76F51",
        "glow_strength": 1.2,
        "motion_style": "pulse"
    },
    # Green phosphor terminal look on near-black; strongest glow (1.5) of the set.
    "Matrix Verdant": {
        "description": "Classic hacker green phosphor terminal layout with black grid backgrounds.",
        "accent": "#33FF33",
        "accent_alt": "#16B816",
        "bg_deep": "#000000",
        "bg_surface": "#051408",
        "bg_raised": "#09210E",
        "border": "#0F3818",
        "border_hi": "#185926",
        "text_hi": "#85FF85",
        "text_mid": "#24B824",
        "text_lo": "#0E4D1D",
        "success": "#33FF33",
        "warning": "#FFFF33",
        "danger": "#FF3333",
        "glow_strength": 1.5,
        "motion_style": "normal"
    },
    # Graphite surfaces with white-blue electric accents.
    "White Lightning Lab": {
        "description": "Graphite black theme accented by sharp, high-intensity white-blue electric sparks.",
        "accent": "#E0F7FA",
        "accent_alt": "#80DEEA",
        "bg_deep": "#121212",
        "bg_surface": "#1E1E1E",
        "bg_raised": "#2D2D2D",
        "border": "#3D3D3D",
        "border_hi": "#5D5D5D",
        "text_hi": "#FFFFFF",
        "text_mid": "#CCCCCC",
        "text_lo": "#888888",
        "success": "#00E676",
        "warning": "#FFD600",
        "danger": "#FF1744",
        "glow_strength": 0.8,
        "motion_style": "pulse"
    },
    # Cyber blue with magenta borders and bright highlights.
    "Neon Circuit": {
        "description": "Cyber blue with high-energy magenta borders and flashing highlights.",
        "accent": "#00E5FF",
        "accent_alt": "#FF007F",
        "bg_deep": "#080312",
        "bg_surface": "#120A2B",
        "bg_raised": "#21124C",
        "border": "#3A1A80",
        "border_hi": "#6124C2",
        "text_hi": "#F0F4FF",
        "text_mid": "#C5A3FF",
        "text_lo": "#7C5EB8",
        "success": "#50FA7B",
        "warning": "#FFB86C",
        "danger": "#FF5555",
        "glow_strength": 1.6,
        "motion_style": "pulse"
    },
    # Icy blue-gray surfaces, restrained glow; the lightest-feeling palette.
    "Arctic Mainframe": {
        "description": "Icy light blue-gray surfaces, clean white panels, and highly restrained glow.",
        "accent": "#88C0D0",
        "accent_alt": "#5E81AC",
        "bg_deep": "#1D232A",
        "bg_surface": "#2E3440",
        "bg_raised": "#3B4252",
        "border": "#434C5E",
        "border_hi": "#4C566A",
        "text_hi": "#ECEFF4",
        "text_mid": "#D8DEE9",
        "text_lo": "#4C566A",
        "success": "#A3BE8C",
        "warning": "#EBCB8B",
        "danger": "#BF616A",
        "glow_strength": 0.4,
        "motion_style": "slow"
    },
    # Near-black console with subtle violet highlights and muted starlight text.
    "Deep Space Console": {
        "description": "Near-black console accented by subtle dark violet highlights and muted starlight text.",
        "accent": "#7B2CBF",
        "accent_alt": "#3C096C",
        "bg_deep": "#030008",
        "bg_surface": "#0A0414",
        "bg_raised": "#18092B",
        "border": "#240E3E",
        "border_hi": "#3B1863",
        "text_hi": "#F3E8FF",
        "text_mid": "#C084FC",
        "text_lo": "#701A75",
        "success": "#4ADE80",
        "warning": "#FBBF24",
        "danger": "#F87171",
        "glow_strength": 0.6,
        "motion_style": "slow"
    },
    # Scientific-instrument look: teal borders, high readability.
    "Quantum Teal": {
        "description": "Precision scientific instrument look with quantum teal borders and high readability.",
        "accent": "#00B4D8",
        "accent_alt": "#0077B6",
        "bg_deep": "#03071E",
        "bg_surface": "#0F1D36",
        "bg_raised": "#1E3E5B",
        "border": "#2A547E",
        "border_hi": "#3F729E",
        "text_hi": "#E0FBFC",
        "text_mid": "#98C1D9",
        "text_lo": "#3D5A80",
        "success": "#06D6A0",
        "warning": "#FFD166",
        "danger": "#EF476F",
        "glow_strength": 1.0,
        "motion_style": "normal"
    },
    # Muted dark base with golden/amber warning-style accents.
    "Solar Flare Dark": {
        "description": "Muted dark base with high-contrast golden flare warnings and solar amber lines.",
        "accent": "#FF9100",
        "accent_alt": "#FF6D00",
        "bg_deep": "#100C08",
        "bg_surface": "#1C1610",
        "bg_raised": "#2A2018",
        "border": "#403024",
        "border_hi": "#5E4635",
        "text_hi": "#FFF3E0",
        "text_mid": "#FFE0B2",
        "text_lo": "#B58E6F",
        "success": "#AEEA00",
        "warning": "#FFD600",
        "danger": "#FF3D00",
        "glow_strength": 1.1,
        "motion_style": "normal"
    },
    # Charcoal and black with tactical crimson alerts (security-console feel).
    "Red Team Ops": {
        "description": "Charcoal and black base layered with tactical crimson alerts for a security console environment.",
        "accent": "#FF1744",
        "accent_alt": "#D50000",
        "bg_deep": "#0D0A0A",
        "bg_surface": "#1A1414",
        "bg_raised": "#2A1E1E",
        "border": "#442424",
        "border_hi": "#663232",
        "text_hi": "#FFEBEE",
        "text_mid": "#FFCDD2",
        "text_lo": "#B71C1C",
        "success": "#00E676",
        "warning": "#FFEA00",
        "danger": "#FF1744",
        "glow_strength": 1.3,
        "motion_style": "normal"
    },
    # Slate 'glass' background with soft glowing borders.
    "Ghost Glass": {
        "description": "Simulated transparent slate glass background with soft glowing borders.",
        "accent": "#00B0FF",
        "accent_alt": "#0091EA",
        "bg_deep": "#0E1216",
        "bg_surface": "#161B22",
        "bg_raised": "#21262D",
        "border": "#30363D",
        "border_hi": "#8B949E",
        "text_hi": "#F0F6FC",
        "text_mid": "#C9D1D9",
        "text_lo": "#8B949E",
        "success": "#34D399",
        "warning": "#FBBF24",
        "danger": "#F87171",
        "glow_strength": 0.7,
        "motion_style": "slow"
    },
    # Very low-glare coding palette: slate panels, indigo highlights.
    "Midnight Compiler": {
        "description": "Extremely low-glare midnight coding view with slate panels and indigo highlights.",
        "accent": "#6366F1",
        "accent_alt": "#4F46E5",
        "bg_deep": "#030712",
        "bg_surface": "#0B1329",
        "bg_raised": "#1E293B",
        "border": "#334155",
        "border_hi": "#475569",
        "text_hi": "#F8FAFC",
        "text_mid": "#94A3B8",
        "text_lo": "#475569",
        "success": "#10B981",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "glow_strength": 0.5,
        "motion_style": "slow"
    },
    # Holographic projection style with thin pale-cyan contours.
    "Hologram Blue": {
        "description": "Futuristic holographic projection interface utilizing thin pale cyan tracing contours.",
        "accent": "#80DEEA",
        "accent_alt": "#4DD0E1",
        "bg_deep": "#040B10",
        "bg_surface": "#0A1B24",
        "bg_raised": "#122B38",
        "border": "#1A3D4F",
        "border_hi": "#2C6B8A",
        "text_hi": "#E0F7FA",
        "text_mid": "#80DEEA",
        "text_lo": "#4D6D7A",
        "success": "#26A69A",
        "warning": "#FFD54F",
        "danger": "#FF8A80",
        "glow_strength": 1.4,
        "motion_style": "pulse"
    },
    # Muted neon violet and magenta tints.
    "Synthwave Control": {
        "description": "Muted neon violet, deep magenta tints, and high-energy tracing circuits.",
        "accent": "#D946EF",
        "accent_alt": "#A21CAF",
        "bg_deep": "#0F051D",
        "bg_surface": "#1A0B2E",
        "bg_raised": "#291147",
        "border": "#4A1E7B",
        "border_hi": "#7E22CE",
        "text_hi": "#FDF4FF",
        "text_mid": "#E879F9",
        "text_lo": "#A21CAF",
        "success": "#4ADE80",
        "warning": "#FACC15",
        "danger": "#F87171",
        "glow_strength": 1.2,
        "motion_style": "normal"
    },
    # Black/gray/white with stark electric-blue indicators; high contrast.
    "Monochrome Signal": {
        "description": "High-contrast black, gray, and white layouts with stark electric blue indicators.",
        "accent": "#2563EB",
        "accent_alt": "#1D4ED8",
        "bg_deep": "#000000",
        "bg_surface": "#171717",
        "bg_raised": "#262626",
        "border": "#404040",
        "border_hi": "#737373",
        "text_hi": "#FFFFFF",
        "text_mid": "#A3A3A3",
        "text_lo": "#525252",
        "success": "#16A34A",
        "warning": "#CA8A04",
        "danger": "#DC2626",
        "glow_strength": 0.9,
        "motion_style": "static"
    },
    # Softer green CRT look tuned for long reading sessions.
    "Emerald Archive": {
        "description": "Softer, historical green CRT archival view optimized for long documentation reads.",
        "accent": "#10B981",
        "accent_alt": "#059669",
        "bg_deep": "#020B06",
        "bg_surface": "#061F12",
        "bg_raised": "#0E3823",
        "border": "#175437",
        "border_hi": "#278A5B",
        "text_hi": "#D1FAE5",
        "text_mid": "#6EE7B7",
        "text_lo": "#1F7A4D",
        "success": "#10B981",
        "warning": "#FBBF24",
        "danger": "#EF4444",
        "glow_strength": 0.8,
        "motion_style": "slow"
    },
    # Slate gray with lightning-blue accents and strong alert borders.
    "Storm Grid": {
        "description": "Slate gray surfaces, electric lightning blue accents, and high-contrast alert borders.",
        "accent": "#60A5FA",
        "accent_alt": "#2563EB",
        "bg_deep": "#0F172A",
        "bg_surface": "#1E293B",
        "bg_raised": "#334155",
        "border": "#475569",
        "border_hi": "#64748B",
        "text_hi": "#F1F5F9",
        "text_mid": "#94A3B8",
        "text_lo": "#475569",
        "success": "#34D399",
        "warning": "#FBBF24",
        "danger": "#F87171",
        "glow_strength": 1.0,
        "motion_style": "normal"
    },
    # Plasma cyan plus neon pink accents; high glow threshold.
    "Plasma Lab": {
        "description": "High-energy plasma cyan mixed with neon pink accents and high glow thresholds.",
        "accent": "#06B6D4",
        "accent_alt": "#FF007F",
        "bg_deep": "#07020E",
        "bg_surface": "#140722",
        "bg_raised": "#26103E",
        "border": "#3E1E62",
        "border_hi": "#61329A",
        "text_hi": "#FAE8FF",
        "text_mid": "#D946EF",
        "text_lo": "#701A75",
        "success": "#10B981",
        "warning": "#FBBF24",
        "danger": "#F43F5E",
        "glow_strength": 1.4,
        "motion_style": "pulse"
    },
    # Charcoal tactical surfaces, low-contrast borders, very low glow.
    "Stealth Operator": {
        "description": "Charcoal tactical surfaces, low contrast borders, and extremely low glow strength.",
        "accent": "#3B82F6",
        "accent_alt": "#1D4ED8",
        "bg_deep": "#0B0F19",
        "bg_surface": "#151D2F",
        "bg_raised": "#1F293D",
        "border": "#2E3B4E",
        "border_hi": "#3E4C5E",
        "text_hi": "#E2E8F0",
        "text_mid": "#94A3B8",
        "text_lo": "#475569",
        "success": "#10B981",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "glow_strength": 0.2,
        "motion_style": "slow"
    },
    # Neon cyan and pure white on pitch black for maximum contrast.
    "God Mode Cyan": {
        "description": "High-intensity neon cyan and pure white accents over pitch black for absolute contrast.",
        "accent": "#00FFFF",
        "accent_alt": "#FFFFFF",
        "bg_deep": "#000000",
        "bg_surface": "#080808",
        "bg_raised": "#141414",
        "border": "#2C2C2C",
        "border_hi": "#4C4C4C",
        "text_hi": "#FFFFFF",
        "text_mid": "#E5E5E5",
        "text_lo": "#888888",
        "success": "#00FF66",
        "warning": "#FFCC00",
        "danger": "#FF2255",
        "glow_strength": 1.8,
        "motion_style": "pulse"
    },
    # Obsidian variant with deeper blacks and cooler borders.
    "Karl Midnight": {
        "description": "Pure midnight variant — same obsidian core but deeper blacks and cooler border tones.",
        "accent": "#00E5FF",
        "accent_alt": "#007A99",
        "bg_deep": "#000000",
        "bg_surface": "#080810",
        "bg_raised": "#0F0F20",
        "border": "#18183A",
        "border_hi": "#28285A",
        "text_hi": "#F0F5FF",
        "text_mid": "#8899BB",
        "text_lo": "#445577",
        "success": "#00FFAA",
        "warning": "#FFCC00",
        "danger": "#FF3366",
        "glow_strength": 1.2,
        "motion_style": "normal"
    },
    # Cooler blue-gray slate base with subdued cyan accents.
    "Karl Slate": {
        "description": "Cooler blue-gray slate base with subdued cyan accents and high-readability text.",
        "accent": "#7EC8E3",
        "accent_alt": "#4A9BB8",
        "bg_deep": "#0A0D11",
        "bg_surface": "#141820",
        "bg_raised": "#1E2430",
        "border": "#2A3040",
        "border_hi": "#3A4254",
        "text_hi": "#DDE3EF",
        "text_mid": "#8A96AA",
        "text_lo": "#505A6E",
        "success": "#5DDBA4",
        "warning": "#E8C96A",
        "danger": "#E06070",
        "glow_strength": 0.6,
        "motion_style": "slow"
    },
    # Warm amber/brown with orange accents for night sessions.
    "Karl Ember": {
        "description": "Warm amber and deep brown tones with glowing orange accents for focused night sessions.",
        "accent": "#FF9A3C",
        "accent_alt": "#CC6A00",
        "bg_deep": "#0C0800",
        "bg_surface": "#1A1205",
        "bg_raised": "#261A08",
        "border": "#3D2A10",
        "border_hi": "#5A3E1A",
        "text_hi": "#FFF2DE",
        "text_mid": "#C8A870",
        "text_lo": "#7A6040",
        "success": "#78C84A",
        "warning": "#FFCC00",
        "danger": "#FF4444",
        "glow_strength": 0.9,
        "motion_style": "normal"
    }
}

# Module-level default palette for code that needs colours without a state object.
# Module-level default palette for code that styles widgets before any AppState
# exists (import-time constants, early splash). Runtime theming should go through
# get_theme_colors() so mode/tone/accent overrides apply.
PALETTE = THEMES["Karl Obsidian Core"]

# CSS font stacks interpolated into the QSS. Order is the fallback order:
# first installed family wins, then the generic family as a last resort.
MONO = (
    "'JetBrains Mono', 'Fira Code', 'Cascadia Code', "
    "'Consolas', 'Courier New', monospace"
)

SANS = (
    "'Inter', 'Segoe UI', 'Ubuntu', 'SF Pro Display', "
    "'Helvetica Neue', 'Arial', sans-serif"
)

def darken_hex_color(hex_str: str, factor: float = 0.7) -> str:
    """Darken a hex color string by a given factor."""
    # Accept '#RRGGBB' or 'RRGGBB'; anything not 6 digits (e.g. a 3-digit '#FFF'
    # shorthand or a CSS name) is not expanded here, it falls back to a dark teal
    # so a bad custom accent can never leave the UI without a visible highlight.
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        return "#008CB8"
    try:
        # int(..., 16) raises ValueError on non-hex digits ('#GGHHII'); caught below.
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        # Scale each channel and clamp to the valid 0-255 byte range.
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return f"#{r:02X}{g:02X}{b:02X}"
    except ValueError:
        return "#008CB8"

def _tint_hex_color(hex_str: str, r_mod: int, g_mod: int, b_mod: int) -> str:
    """Tint a hex color by adding offsets to its RGB channels."""
    # Malformed input is returned unchanged (re-prefixed with '#') rather than
    # replaced with a default: callers use this for subtle shading, where an
    # untouched colour is a better failure mode than a wrong one.
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        return "#" + hex_str
    try:
        # Offsets are signed ints and the result is clamped to 0-255 per channel.
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        r = max(0, min(255, r + r_mod))
        g = max(0, min(255, g + g_mod))
        b = max(0, min(255, b + b_mod))
        return f"#{r:02X}{g:02X}{b:02X}"
    except ValueError:
        return "#" + hex_str

def hex_to_rgba(hex_str: str, alpha: float) -> str:
    """Convert a hex color string to rgba format."""
    # QSS has no alpha-hex support in every Qt version, so translucent colours
    # are emitted as rgba(). alpha is 0.0-1.0 (QSS accepts a float here).
    # The fallback is the stock cyan so glass panels stay visible on bad input.
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        return f"rgba(0, 194, 255, {alpha})"
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    except ValueError:
        return f"rgba(0, 194, 255, {alpha})"

def get_theme_colors(state_or_name, custom_accent=None, bg_tone="Default", mode: str = None) -> dict:
    """Get the palette colors dictionary for the given theme, custom accent and background overlays."""
    # Two call shapes exist: the modern one passes the AppState (theme, accent and
    # mode all live on it), the legacy one passes a theme name plus loose kwargs.
    # State object: read everything from it (tone is not stored on state).
    if hasattr(state_or_name, "theme_preset"):
        state = state_or_name
        theme_name = getattr(state, "theme_preset", "Karl Obsidian Core")
        custom_accent = getattr(state, "custom_accent", None)
        if mode is None:
            mode = getattr(state, "theme_mode", "midnight")
        tone = "Default"
    else:
        theme_name = state_or_name
        # Legacy name from older saved configs.
        if theme_name == "Karl Obsidian":
            theme_name = "Karl Obsidian Core"
        if mode is None:
            mode = "midnight"
        tone = bg_tone
        
    # Copy so the per-call overrides below never mutate the shared registry.
    raw = dict(THEMES.get(theme_name, THEMES["Karl Obsidian Core"]))

    # Modes re-skin the backgrounds of any theme; 'ember' also swaps the accent
    # to orange unless the user picked a custom one.
    # Theme mode overrides
    # Modes only swap background/border ramps; text and status colours stay as the
    # theme defines them so contrast ratios the theme author tuned are preserved.
    # "midnight" is the identity mode (no override), which is why it has no branch.
    if mode == "slate":
        raw["bg_deep"] = "#0B0F17"
        raw["bg_surface"] = "#151C2C"
        raw["bg_raised"] = "#202A3E"
        raw["border"] = "#2E364A"
        raw["border_hi"] = "#3E4C66"
    elif mode == "ember":
        raw["bg_deep"] = "#120C0A"
        raw["bg_surface"] = "#1A1310"
        raw["bg_raised"] = "#261D1A"
        raw["border"] = "#3D2E29"
        raw["border_hi"] = "#5C4740"
        if not custom_accent:
            raw["accent"] = "#FF8C00"
            raw["accent_alt"] = "#B35F00"

    # A user-chosen accent wins; its alt shade is derived by darkening 30%.
    if custom_accent:
        raw["accent"] = custom_accent
        raw["accent_alt"] = darken_hex_color(custom_accent, 0.7)
    else:
        raw["accent_alt"] = raw.get("accent_alt", darken_hex_color(raw["accent"], 0.7))
        
    # Tone is a legacy per-call argument (AppState never carries it), kept so older
    # callers and saved configs still resolve to the same colours.
    # Background tones are applied last so they win over mode overrides.
    # Tone overrides
    if tone == "Pitch Black":
        raw["bg_deep"] = "#000000"
        raw["bg_surface"] = "#080808"
        raw["bg_raised"] = "#141414"
    elif tone == "Warm Sepia":
        raw["bg_deep"] = "#120F0D"
        raw["bg_surface"] = "#1A1614"
        raw["bg_raised"] = "#26201D"
    elif tone == "Cool Slate":
        raw["bg_deep"] = "#0B0F17"
        raw["bg_surface"] = "#151C2C"
        raw["bg_raised"] = "#202A3E"
        
    # Flatten into the key names the QSS template uses. bg_input and sidebar/think
    # shades are derived by darkening so they stay consistent for any theme.
    p = {
        "bg_deep":     raw["bg_deep"],
        "bg_base":     raw["bg_deep"],
        "bg_surface":  raw["bg_surface"],
        "bg_raised":   raw["bg_raised"],
        "bg_input":    darken_hex_color(raw["bg_surface"], 0.7),
        "border":      raw["border"],
        "border_hi":   raw["border_hi"],
        "accent":      raw["accent"],
        "accent_dark": raw["accent_alt"],
        "text_hi":     raw["text_hi"],
        "text_mid":    raw["text_mid"],
        "text_lo":     raw["text_lo"],
        "think_bg":    darken_hex_color(raw["bg_deep"], 0.8),
        "think_text":  raw["text_mid"],
        "green":       raw["success"],
        "red":         raw["danger"],
        "yellow":      raw["warning"],
        "sidebar_bg":  darken_hex_color(raw["bg_deep"], 0.85),
        "sidebar_sel": darken_hex_color(raw["bg_raised"], 0.9),
        
        # Translucent variants for the glass/HUD styling; alpha values are tuned
        # by eye (0.75 panels, 0.15 borders, 0.3 glow).
        # Glassmorphism / Heavenscape extensions
        "bg_surface_glass": hex_to_rgba(raw["bg_surface"], 0.75),
        "bg_raised_glass":  hex_to_rgba(raw["bg_raised"], 0.75),
        "border_glass":     hex_to_rgba(raw["accent"], 0.15),
        "accent_glow":      hex_to_rgba(raw["accent"], 0.3),
        "idle_border":      hex_to_rgba(raw["accent"], 0.8),
        "generating_border": "rgba(255, 140, 0, 0.8)",
        "error_border":     "rgba(255, 59, 48, 0.9)",
    }
    
    # Non-colour metadata rides along in the same dict so the appearance panel can
    # read description/glow/motion without a second registry lookup. Defaults keep
    # third-party/legacy theme entries that omit these keys working.
    p["description"] = raw.get("description", "")
    p["glow_strength"] = raw.get("glow_strength", 1.0)
    p["motion_style"] = raw.get("motion_style", "normal")
    return p

# Compiled QSS memo: the sheet is a pure function of (theme, accent, tone,
# layout), so repeat applications (live preview, startup, slider edits) reuse
# the cached string instead of re-formatting the ~450-line template.
# Bounded: cleared wholesale when it reaches the max rather than tracking LRU,
# which is fine because only a handful of distinct combinations occur.
_STYLESHEET_CACHE: dict = {}
_STYLESHEET_CACHE_MAX = 64


def get_theme_stylesheet(state_or_name, custom_accent=None, bg_tone="Default", mode: str = None) -> str:
    """Compile the QSS stylesheet for the given state configurations."""
    # The cache key must include every input that changes the output, including
    # layout_preset and mode, or switching them would serve a stale sheet.
    # Layout preset only affects spacing/font size, not colours.
    layout_preset = "Focused Workbench"
    if hasattr(state_or_name, "layout_preset"):
        layout_preset = getattr(state_or_name, "layout_preset", "Focused Workbench")

    theme_mode = "midnight"
    if hasattr(state_or_name, "theme_mode"):
        theme_mode = getattr(state_or_name, "theme_mode", "midnight")
    elif mode is not None:
        theme_mode = mode

    if hasattr(state_or_name, "theme_preset"):
        cache_key = (
            getattr(state_or_name, "theme_preset", "Karl Obsidian Core"),
            getattr(state_or_name, "custom_accent", None),
            "Default",
            layout_preset,
            theme_mode,
        )
    else:
        cache_key = (state_or_name, custom_accent, bg_tone, layout_preset, theme_mode)

    # Cache hit: skip re-formatting the large template.
    cached = _STYLESHEET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    p = get_theme_colors(state_or_name, custom_accent, bg_tone, mode=theme_mode)
    p["mono"] = MONO
    p["sans"] = SANS

    # Default density; presets below tighten or loosen margins, spacing and font.
    margin = 12
    spacing = 10
    font_size = 10
    
    if layout_preset == "Compact Laptop":
        margin = 4
        spacing = 4
        font_size = 9
    elif layout_preset == "Wide Monitor Command":
        margin = 20
        spacing = 16
        font_size = 11
    elif layout_preset == "Minimal Distraction":
        margin = 8
        spacing = 6
        font_size = 10
        
    p["margin"] = str(margin)
    p["spacing"] = str(spacing)
    p["font_size"] = str(font_size)

    # str.format fills the {placeholders} in _SHEET; literal QSS braces are
    # doubled ({{ }}) in the template for that reason.
    sheet = _SHEET.format(**p)
    if len(_STYLESHEET_CACHE) >= _STYLESHEET_CACHE_MAX:
        _STYLESHEET_CACHE.clear()
    _STYLESHEET_CACHE[cache_key] = sheet
    return sheet

# QSS template (a plain string, so no comments can live inside it). Sections:
# reset, text widgets, buttons (primary/danger/success/warning/ghost/icon),
# combo boxes, sliders, spin boxes, lists, check boxes, group boxes, docks,
# progress bars, tooltips, scrollbars and the HUD/settings overlays. Colours,
# fonts and spacing come from get_theme_colors() via str.format.
_SHEET = """\
/* ── Reset ───────────────────────────────────────────────── */
* {{ outline: none; }}

QMainWindow, QDialog {{
    background: {bg_deep};
}}

QWidget {{
    background: transparent;
    color: {text_hi};
    font-family: {sans};
    font-size: {font_size}pt;
    border: none;
}}

QTextBrowser, QTextEdit, QPlainTextEdit, #reasoning-view, #code-preview {{
    font-family: {mono};
}}

/* ── Sidebar ──────────────────────────────────────────────── */
#sidebar {{
    background: {sidebar_bg};
    border-right: 1px solid {border};
}}

#sidebar-logo {{
    color: {accent};
    font-size: 15pt;
    font-weight: bold;
    background: transparent;
    padding: 0;
}}

#sidebar-btn {{
    background: transparent;
    border: none;
    border-radius: 6px;
    color: {text_lo};
    padding: 6px 2px;
    font-size: 8pt;
    text-align: center;
    width: 52px;
}}

#sidebar-btn:hover {{
    background: {bg_raised};
    color: {text_mid};
}}

#sidebar-btn[active="true"] {{
    background: {sidebar_sel};
    color: {accent};
    border-left: 2px solid {accent};
    border-radius: 0 6px 6px 0;
}}

/* ── Workspace Shell ──────────────────────────────────────── */
#workspace-root {{
    background: {bg_base};
    padding: {margin}px;
    border: 1px solid transparent;
}}

#panel {{
    background: {bg_surface_glass};
    border: 1px solid {border_glass};
    border-radius: 4px;
}}

#panel-header {{
    background: {bg_raised_glass};
    border-bottom: 1px solid {border_glass};
    border-radius: 4px 4px 0 0;
    padding: 5px 12px;
    color: {text_lo};
    font-size: 8pt;
    letter-spacing: 2px;
}}

/* State-driven QSS styles for dynamic glow borders */
#panel[modelState="idle"], #workspace-root[modelState="idle"], #settings-overlay[modelState="idle"] {{
    border-color: {accent};
}}

#panel[modelState="generating"], #workspace-root[modelState="generating"], #settings-overlay[modelState="generating"] {{
    border-color: #FF9F00;
}}

#panel[modelState="error"], #workspace-root[modelState="error"], #settings-overlay[modelState="error"] {{
    border-color: #FF3B30;
}}

/* ── Status Bar ───────────────────────────────────────────── */
#status-bar {{
    background: {bg_surface};
    border-top: 1px solid {border};
}}

/* ── Label Roles ──────────────────────────────────────────── */
#lbl-muted {{
    color: {text_lo};
}}

#lbl-mid {{
    color: {text_mid};
}}

#lbl-accent {{
    color: {accent};
}}

#lbl-danger, #lbl-red {{
    color: {red};
}}

#lbl-warning {{
    color: {yellow};
}}

#lbl-green, #lbl-success {{
    color: {green};
}}

/* ── Text Displays ────────────────────────────────────────── */
QTextBrowser {{
    background: transparent;
    border: none;
    color: {text_hi};
    font-family: {mono};
    font-size: {font_size}pt;
    selection-background-color: {accent_dark};
    padding: 6px;
}}

#reasoning-view {{
    background: {think_bg};
    color: {think_text};
    font-size: 9pt;
    border-right: 1px solid {border};
}}

/* ── Inputs ───────────────────────────────────────────────── */
QTextEdit {{
    background: {bg_input};
    border: 1px solid {border};
    border-radius: 4px;
    color: {text_hi};
    font-family: {mono};
    font-size: {font_size}pt;
    padding: 8px 10px;
    selection-background-color: {accent_dark};
}}

QTextEdit:focus {{
    border-color: {accent};
}}

QLineEdit {{
    background: {bg_input};
    border: 1px solid {border};
    border-radius: 4px;
    color: {text_hi};
    font-family: {mono};
    font-size: {font_size}pt;
    padding: 6px 10px;
    selection-background-color: {accent_dark};
}}

QLineEdit:focus {{
    border-color: {accent};
}}

QLineEdit:read-only {{
    color: {text_mid};
    background: {bg_surface};
}}

/* ── Buttons ──────────────────────────────────────────────── */
QPushButton {{
    background: {bg_raised};
    border: 1px solid {border};
    border-radius: 4px;
    color: {text_mid};
    padding: 6px 14px;
    font-family: {sans};
    font-size: {font_size}pt;
}}

QPushButton:hover {{
    background: {border};
    color: {text_hi};
    border-color: {border_hi};
}}

QPushButton:pressed {{
    background: {bg_surface};
}}

QPushButton:disabled {{
    color: {text_lo};
    border-color: {border};
    background: {bg_surface};
}}

QPushButton#btn-primary {{
    background: {accent};
    border: none;
    color: #050510;
    font-weight: bold;
    padding: 6px 20px;
}}

QPushButton#btn-primary:hover {{
    background: {accent_dark};
    color: #050510;
}}

QPushButton#btn-primary:disabled {{
    background: {border};
    color: {text_lo};
}}

QPushButton#btn-danger {{
    background: transparent;
    border: 1px solid {red};
    color: {red};
}}

QPushButton#btn-danger:hover {{
    background: {red};
    color: #ffffff;
}}

QPushButton#btn-success {{
    background: transparent;
    border: 1px solid {green};
    color: {green};
    border-radius: 4px;
}}

QPushButton#btn-success:hover {{
    background: {green};
    color: #050510;
    font-weight: bold;
}}

QPushButton#btn-warning {{
    background: transparent;
    border: 1px solid {yellow};
    color: {yellow};
    border-radius: 4px;
}}

QPushButton#btn-warning:hover {{
    background: {yellow};
    color: #050510;
    font-weight: bold;
}}

QPushButton#btn-ghost {{
    background: transparent;
    border: none;
    color: {text_lo};
    padding: 4px 8px;
    font-size: 9pt;
}}

QPushButton#btn-ghost:hover {{
    color: {text_mid};
    background: transparent;
}}

QPushButton#btn-ghost:disabled {{
    color: {text_lo};
    background: transparent;
    border: none;
}}

QPushButton#btn-icon {{
    background: transparent;
    border: none;
    color: {text_lo};
    padding: 2px;
    font-size: 11pt;
}}

QPushButton#btn-icon:hover {{
    color: {text_mid};
}}

/* ── ComboBox ─────────────────────────────────────────────── */
QComboBox {{
    background: {bg_input};
    border: 1px solid {border};
    border-radius: 4px;
    color: {text_hi};
    padding: 5px 10px;
    font-family: {sans};
    font-size: {font_size}pt;
}}

QComboBox:hover {{
    border-color: {border_hi};
}}

QComboBox:focus {{
    border-color: {accent};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
    padding-right: 4px;
}}

QComboBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {text_lo};
}}

QComboBox QAbstractItemView {{
    background: {bg_raised};
    border: 1px solid {border_hi};
    color: {text_hi};
    selection-background-color: {accent_dark};
    outline: none;
    padding: 2px;
}}

/* ── Sliders ──────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 2px;
    background: {border};
    border-radius: 1px;
}}

QSlider::handle:horizontal {{
    background: {accent};
    border: none;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}

QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 1px;
}}

/* ── SpinBox ──────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background: {bg_input};
    border: 1px solid {border};
    border-radius: 4px;
    color: {text_hi};
    padding: 4px 8px;
    font-family: {mono};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {accent};
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {bg_raised};
    border: none;
    width: 18px;
}}

/* ── ListWidget ───────────────────────────────────────────── */
QListWidget {{
    background: {bg_surface};
    border: 1px solid {border};
    border-radius: 6px;
    color: {text_hi};
    font-family: {mono};
    outline: none;
    padding: 4px;
}}

QListWidget::item {{
    padding: 8px 12px;
    border-radius: 4px;
    margin: 2px 4px;
}}

QListWidget::item:hover {{
    background: {bg_raised};
    color: {text_hi};
}}

QListWidget::item:selected {{
    background: {sidebar_sel};
    color: {accent};
    font-weight: bold;
}}

/* ── CheckBox ─────────────────────────────────────────────── */
QCheckBox {{
    color: {text_hi};
    spacing: 8px;
    font-family: {mono};
}}

QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {border_hi};
    border-radius: 3px;
    background: {bg_input};
}}

QCheckBox::indicator:checked {{
    background: {accent};
    border-color: {accent};
}}

/* ── GroupBox ─────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {border};
    border-radius: 4px;
    margin-top: 14px;
    padding-top: 8px;
    color: {text_lo};
    font-size: 8pt;
    letter-spacing: 1px;
    font-family: {mono};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    background: {bg_base};
}}

/* ── Docks & Splitters ────────────────────────────────────── */
QDockWidget {{
    background: {bg_surface};
    border: 1px solid {border};
    border-radius: 4px;
}}

QDockWidget::title {{
    background: {bg_raised};
    border-bottom: 1px solid {border};
    padding: 6px 12px;
    color: {text_hi};
    font-size: 8pt;
    font-weight: bold;
    letter-spacing: 2px;
    text-align: left;
}}

QDockWidget::close-button, QDockWidget::float-button {{
    border: none;
    background: transparent;
    padding: 2px;
    border-radius: 3px;
}}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background: {border};
}}

QMainWindow::separator {{
    background: {border};
    width: 2px;
    height: 2px;
}}

QMainWindow::separator:hover {{
    background: {accent};
}}

/* ── Token Budget Bar ──────────────────────────────────────── */
QProgressBar#token-budget-bar {{
    background: {bg_input};
    border: none;
    border-radius: 3px;
}}

/* ── Workbench Chat Composer ─────────────────────────────── */
#chat-composer {{
    background: {bg_surface};
    border-top: 1px solid {border_hi};
}}

#token-row {{
    background: {bg_surface};
}}

#chat-composer QTextEdit {{
    background: {bg_input};
    border: 1px solid {border};
    border-radius: 5px;
    padding: 8px 10px;
    color: {text_hi};
    selection-background-color: {accent};
}}

#chat-composer QComboBox {{
    min-height: 26px;
}}

#workspace-root[responsiveMode="focus"] #hud-toolbar,
#workspace-root[responsiveMode="single"] #hud-toolbar {{
    padding: 4px 8px;
}}

#workspace-root[responsiveMode="focus"] #hud-btn,
#workspace-root[responsiveMode="single"] #hud-btn {{
    padding: 3px 7px;
    font-size: 8pt;
}}

#workspace-root[responsiveMode="single"] #chat-composer QTextEdit {{
    padding: 6px 8px;
}}

/* ── Reload Notice Banner ────────────────────────────────── */
#reload-notice {{
    background: {bg_surface};
    color: {accent};
    border-bottom: 1px solid {border};
    font-size: 8.5pt;
}}

/* ── ToolTip ──────────────────────────────────────────────── */
QToolTip {{
    background: {bg_raised};
    border: 1px solid {border_hi};
    color: {text_hi};
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 9pt;
    font-family: {mono};
}}

/* ── Scrollbars ────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {accent_glow};
    min-height: 20px;
    border-radius: 3px;
}}

QScrollBar::handle:vertical:hover {{
    background: {accent};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: transparent;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
    margin: 0px;
}}

QScrollBar::handle:horizontal {{
    background: {accent_glow};
    min-width: 20px;
    border-radius: 3px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {accent};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    background: transparent;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* ── HUD Toolbar ─────────────────────────────────────────── */
#hud-toolbar {{
    background: {bg_surface_glass};
    border-bottom: 1px solid {border_glass};
    padding: 6px 12px;
}}

#hud-btn {{
    background: transparent;
    border: 1px solid {border_glass};
    border-radius: 4px;
    color: {text_mid};
    padding: 4px 10px;
    font-size: 8.5pt;
    font-family: {sans};
}}

#hud-btn:hover {{
    background: {bg_raised_glass};
    border-color: {accent};
    color: {text_hi};
}}

#hud-btn[active="true"] {{
    background: {sidebar_sel};
    border-color: {accent};
    color: {accent};
    font-weight: bold;
}}

/* ── Settings Overlay ─────────────────────────────────────── */
#settings-overlay {{
    background: rgba(16, 20, 30, 0.95);
    border: 1px solid {border_glass};
    border-radius: 8px;
}}

#settings-overlay-header {{
    background: {bg_raised_glass};
    border-bottom: 1px solid {border_glass};
    border-radius: 8px 8px 0 0;
    padding: 6px 12px;
    color: {accent};
    font-size: 8.5pt;
    font-weight: bold;
    letter-spacing: 2px;
    font-family: {mono};
}}

#settings-overlay QLabel {{
    font-size: 8.5pt;
}}
"""

def stylesheet(accent: str = ACCENT_DEFAULT, mode: str = "midnight") -> str:
    """Backward-compatible helper returning QSS for the default theme."""
    # Minimal stand-in exposing the attributes get_theme_stylesheet reads, so
    # legacy callers need not construct a full AppState.
    class DummyState:
        theme_preset = "Karl Obsidian Core"
        custom_accent = accent
        layout_preset = "Focused Workbench"
        theme_mode = mode
    return get_theme_stylesheet(DummyState())
