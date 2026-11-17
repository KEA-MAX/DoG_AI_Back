const { createApp } = require('./app');

const PORT = Number(process.env.PORT) || 3000;
const app = createApp();

app.listen(PORT, () => {
  console.log(`DoG_AI_Back temporary backend is running on port ${PORT}`);
});
