class WebSocketClient {
    constructor(url) {
      this.url = url;
      this.connect();
    }
  
    connect() {
      this.socket = new WebSocket(this.url);
  
      this.socket.onopen = () => {
        console.log('WebSocket connection established');
      };
  
      this.socket.onmessage = (event) => {
        console.log('Message received:', event.data);
      };
  
      this.socket.onclose = (event) => {
        console.log('WebSocket connection closed:', event);
        // Reconnect logic if needed
      };
  
      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    }
  }
  
  // Usage
  const wsClient = new WebSocketClient('wss://cautious-space-system-q77pv47r9vrc647g-3000.app.github.dev:3000/ws');
  