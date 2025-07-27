#ifndef UI_MANAGER_H
#define UI_MANAGER_H

#include <string>
#include <memory>

namespace aisis {

class UIManager {
public:
    UIManager() = default;
    ~UIManager() = default;
    
    bool initialize();
    void shutdown();
    void update();
    void render();
    
    // Basic UI functionality
    void showWindow(const std::string& title);
    void hideWindow();
    bool isWindowVisible() const { return window_visible_; }
    
private:
    bool initialized_ = false;
    bool window_visible_ = false;
};

} // namespace aisis

#endif // UI_MANAGER_H