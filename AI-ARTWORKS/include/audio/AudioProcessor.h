/**
 * @file AudioProcessor.h
 * @brief Audio processing system for AI-Artworks
 * @author AI-Artworks Team
 * @version 1.0.0
 */

#pragma once

#include <memory>

class AudioProcessor {
public:
    AudioProcessor();
    ~AudioProcessor();

    bool initialize();
    void shutdown();
    void update(float deltaTime);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};