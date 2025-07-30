/**
 * @file GraphicsEngine.h
 * @brief Graphics rendering engine for AI-Artworks
 * @author AI-Artworks Team
 * @version 1.0.0
 */

#pragma once

#include <memory>
#include <string>

class GraphicsEngine {
public:
    GraphicsEngine();
    ~GraphicsEngine();

    bool initialize();
    void shutdown();
    void update(float deltaTime);
    
    void beginFrame();
    void render();
    void endFrame();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};