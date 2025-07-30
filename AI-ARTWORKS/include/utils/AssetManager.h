/**
 * @file AssetManager.h
 * @brief Asset management system for AI-Artworks
 * @author AI-Artworks Team
 * @version 1.0.0
 */

#pragma once

#include <memory>
#include <string>

class AssetManager {
public:
    AssetManager();
    ~AssetManager();

    bool initialize();
    void shutdown();
    void update(float deltaTime);

    bool loadAsset(const std::string& path);
    void unloadAsset(const std::string& path);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};