const express = require('express');
const app = express();

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*'); // Allows all origins
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

// Example route
app.get('/api/example', (req, res) => {
  res.json({ message: 'This is an example route' });
});

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
