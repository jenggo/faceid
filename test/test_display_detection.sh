#!/bin/bash
# Display Detection Diagnostic Script
# Run this while screen is ON, then again after locking screen

echo "==================================================================="
echo "DISPLAY DETECTION DIAGNOSTIC TOOL"
echo "==================================================================="
echo ""
echo "Run this script in two scenarios:"
echo "1. With screen ON and UNLOCKED"
echo "2. After locking screen (Meta+L) - run from SSH or another terminal"
echo ""
echo "==================================================================="
echo ""

echo "📊 Environment Variables:"
echo "  WAYLAND_DISPLAY: ${WAYLAND_DISPLAY:-<not set>}"
echo "  XDG_SESSION_TYPE: ${XDG_SESSION_TYPE:-<not set>}"
echo "  DISPLAY: ${DISPLAY:-<not set>}"
echo "  PAM_SERVICE: ${PAM_SERVICE:-<not set>}"
echo "  PAM_TTY: ${PAM_TTY:-<not set>}"
echo "  DESKTOP_SESSION: ${DESKTOP_SESSION:-<not set>}"
echo ""

echo "==================================================================="
echo "🔍 Method 1: Systemd Session LockedHint"
echo "==================================================================="
LOCKED_HINT=$(loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p LockedHint --value 2>/dev/null)
echo "  Result: ${LOCKED_HINT:-<empty>}"
if [ "$LOCKED_HINT" = "yes" ] || [ "$LOCKED_HINT" = "true" ] || [ "$LOCKED_HINT" = "1" ]; then
    echo "  ✅ LOCKED"
else
    echo "  ❌ UNLOCKED (or unknown)"
fi
echo ""

echo "==================================================================="
echo "🔍 Method 2: KDE Lock Screen Greeter Process"
echo "==================================================================="
GREETER=$(ps aux | grep -w '[k]screenlocker_greet' 2>/dev/null)
if [ -n "$GREETER" ]; then
    echo "  ✅ FOUND: kscreenlocker_greet is running"
    echo "  Process: $GREETER"
else
    echo "  ❌ NOT FOUND: kscreenlocker_greet is not running"
fi
echo ""

echo "==================================================================="
echo "🔍 Method 3: Backlight Actual Brightness"
echo "==================================================================="
if [ -d "/sys/class/backlight" ]; then
    for bl in /sys/class/backlight/*/actual_brightness; do
        if [ -f "$bl" ]; then
            BRIGHTNESS=$(cat "$bl" 2>/dev/null)
            MAX_BRIGHTNESS=$(cat "${bl%/*}/max_brightness" 2>/dev/null)
            echo "  $(basename $(dirname $bl)): $BRIGHTNESS / $MAX_BRIGHTNESS"
            if [ "$BRIGHTNESS" = "0" ]; then
                echo "    ✅ BACKLIGHT OFF (brightness = 0)"
            else
                PERCENT=$((100 * BRIGHTNESS / MAX_BRIGHTNESS))
                echo "    ❌ BACKLIGHT ON (${PERCENT}%)"
            fi
        fi
    done
else
    echo "  ❌ /sys/class/backlight not found"
fi
echo ""

echo "==================================================================="
echo "🔍 Method 4: DRM DPMS State (eDP only - laptop screen)"
echo "==================================================================="
if ls /sys/class/drm/card*/card*-eDP-*/dpms 2>/dev/null | head -1 > /dev/null; then
    for dpms in /sys/class/drm/card*/card*-eDP-*/dpms; do
        if [ -f "$dpms" ]; then
            DPMS_STATE=$(cat "$dpms" 2>/dev/null)
            echo "  $(basename $(dirname $dpms)): $DPMS_STATE"
            if [ "$DPMS_STATE" = "Off" ] || [ "$DPMS_STATE" = "off" ]; then
                echo "    ✅ DISPLAY OFF"
            else
                echo "    ❌ DISPLAY ON"
            fi
        fi
    done
else
    echo "  ❌ eDP display not found in /sys/class/drm/"
fi
echo ""

echo "==================================================================="
echo "🔍 Method 5: All DRM Displays (for reference)"
echo "==================================================================="
if ls /sys/class/drm/card*/card*-*/dpms 2>/dev/null | head -1 > /dev/null; then
    for dpms in /sys/class/drm/card*/card*-*/dpms; do
        if [ -f "$dpms" ]; then
            DPMS_STATE=$(cat "$dpms" 2>/dev/null)
            CONNECTOR=$(basename $(dirname $dpms))
            echo "  $CONNECTOR: $DPMS_STATE"
        fi
    done
else
    echo "  ❌ No DRM displays found"
fi
echo ""

echo "==================================================================="
echo "🔍 Method 6: X11 DPMS (if X11)"
echo "==================================================================="
if [ -n "$DISPLAY" ] && command -v xset &> /dev/null; then
    XSET_OUTPUT=$(DISPLAY=:0 xset q 2>/dev/null | grep 'Monitor is')
    if [ -n "$XSET_OUTPUT" ]; then
        echo "  $XSET_OUTPUT"
        if echo "$XSET_OUTPUT" | grep -qi "off"; then
            echo "    ✅ DISPLAY OFF"
        else
            echo "    ❌ DISPLAY ON"
        fi
    else
        echo "  ❌ Cannot get DPMS state from xset"
    fi
else
    echo "  ❌ Not X11 or xset not available"
fi
echo ""

echo "==================================================================="
echo "🔍 Method 7: Wayland Idle Inhibit (if available)"
echo "==================================================================="
if [ "$XDG_SESSION_TYPE" = "wayland" ]; then
    if command -v busctl &> /dev/null; then
        IDLE=$(busctl --user get-property org.freedesktop.ScreenSaver /org/freedesktop/ScreenSaver org.freedesktop.ScreenSaver Active 2>/dev/null)
        if [ -n "$IDLE" ]; then
            echo "  ScreenSaver Active: $IDLE"
        else
            echo "  ❌ Could not query ScreenSaver D-Bus"
        fi
    else
        echo "  ❌ busctl not available"
    fi
else
    echo "  ⏭️  Skipped (not Wayland)"
fi
echo ""

echo "==================================================================="
echo "🔍 Method 8: KDE Screen Locker D-Bus"
echo "==================================================================="
if command -v qdbus &> /dev/null; then
    LOCKED=$(qdbus org.freedesktop.ScreenSaver /ScreenSaver org.freedesktop.ScreenSaver.GetActive 2>/dev/null)
    if [ -n "$LOCKED" ]; then
        echo "  ScreenSaver Active: $LOCKED"
        if [ "$LOCKED" = "true" ]; then
            echo "    ✅ LOCKED"
        else
            echo "    ❌ UNLOCKED"
        fi
    else
        echo "  ❌ Could not query KDE ScreenSaver D-Bus"
    fi
elif command -v qdbus6 &> /dev/null; then
    LOCKED=$(qdbus6 org.freedesktop.ScreenSaver /ScreenSaver org.freedesktop.ScreenSaver.GetActive 2>/dev/null)
    if [ -n "$LOCKED" ]; then
        echo "  ScreenSaver Active: $LOCKED"
        if [ "$LOCKED" = "true" ]; then
            echo "    ✅ LOCKED"
        else
            echo "    ❌ UNLOCKED"
        fi
    else
        echo "  ❌ Could not query KDE ScreenSaver D-Bus"
    fi
else
    echo "  ❌ qdbus/qdbus6 not available"
fi
echo ""

echo "==================================================================="
echo "🔍 Method 9: Logind IdleHint"
echo "==================================================================="
IDLE_HINT=$(loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p IdleHint --value 2>/dev/null)
echo "  Result: ${IDLE_HINT:-<empty>}"
if [ "$IDLE_HINT" = "yes" ] || [ "$IDLE_HINT" = "true" ] || [ "$IDLE_HINT" = "1" ]; then
    echo "  ℹ️  Session is IDLE"
else
    echo "  ℹ️  Session is ACTIVE"
fi
echo ""

echo "==================================================================="
echo "📝 SUMMARY"
echo "==================================================================="
echo ""
echo "Test this script in BOTH scenarios:"
echo "  1. Screen ON + UNLOCKED:  bash test_display_detection.sh > unlocked.txt"
echo "  2. Screen LOCKED (Meta+L): bash test_display_detection.sh > locked.txt"
echo ""
echo "Then compare: diff -y unlocked.txt locked.txt"
echo ""
echo "Look for methods that show ✅ when locked but ❌ when unlocked."
echo "Those are the reliable methods for detecting screen-off state!"
echo ""
echo "==================================================================="
