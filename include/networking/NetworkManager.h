#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include <string>
#include <memory>

namespace aisis {

class NetworkManager {
public:
    NetworkManager() = default;
    ~NetworkManager() = default;
    
    bool initialize();
    void shutdown();
    
    bool isConnected() const { return connected_; }
    
    // Basic networking functionality
    bool connect(const std::string& address, int port);
    void disconnect();
    
    bool sendData(const std::string& data);
    std::string receiveData();
    
private:
    bool connected_ = false;
};

} // namespace aisis

#endif // NETWORK_MANAGER_H