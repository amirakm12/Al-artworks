#pragma once

#include <vector>
#include <string>
#include "UltimateToolEngine.h"

namespace AI_ARTWORKS {

class RealityTools {
public:
    // Augmented Reality Tools
    static void arRenderer(const std::vector<std::string>& params);
    static void arTracker(const std::vector<std::string>& params);
    static void arOverlay(const std::vector<std::string>& params);
    static void arInteraction(const std::vector<std::string>& params);
    static void arSpatialMapping(const std::vector<std::string>& params);
    static void arObjectRecognition(const std::vector<std::string>& params);
    static void arGestureControl(const std::vector<std::string>& params);
    static void arVoiceControl(const std::vector<std::string>& params);
    static void arHapticFeedback(const std::vector<std::string>& params);
    static void arEnvironmentalUnderstanding(const std::vector<std::string>& params);

    // Virtual Reality Tools
    static void vrRenderer(const std::vector<std::string>& params);
    static void vrController(const std::vector<std::string>& params);
    static void vrHeadset(const std::vector<std::string>& params);
    static void vrEnvironment(const std::vector<std::string>& params);
    static void vrAvatar(const std::vector<std::string>& params);
    static void vrTeleportation(const std::vector<std::string>& params);
    static void vrHandTracking(const std::vector<std::string>& params);
    static void vrEyeTracking(const std::vector<std::string>& params);
    static void vrHapticSystem(const std::vector<std::string>& params);
    static void vrAudioSpatialization(const std::vector<std::string>& params);

    // Mixed Reality Tools
    static void mrBlender(const std::vector<std::string>& params);
    static void mrPassthrough(const std::vector<std::string>& params);
    static void mrSpatialAnchor(const std::vector<std::string>& params);
    static void mrHologram(const std::vector<std::string>& params);
    static void mrGestureRecognition(const std::vector<std::string>& params);
    static void mrVoiceCommand(const std::vector<std::string>& params);
    static void mrEyeGaze(const std::vector<std::string>& params);
    static void mrHandGesture(const std::vector<std::string>& params);
    static void mrSpatialAudio(const std::vector<std::string>& params);

    // Tool Registration
    static void registerAllTools(UltimateToolEngine& engine);
};

} // namespace AI_ARTWORKS 