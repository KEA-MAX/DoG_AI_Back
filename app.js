const http = require('http');

const mockDogs = [
  { id: 1, name: 'Mugi', breed: 'Shiba Inu', age: 4 },
  { id: 2, name: 'Sora', breed: 'Toy Poodle', age: 2 }
];

function writeJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8'
  });
  res.end(JSON.stringify(payload));
}

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';

    req.on('data', (chunk) => {
      body += chunk;
    });

    req.on('end', () => {
      if (!body) {
        resolve({});
        return;
      }

      try {
        resolve(JSON.parse(body));
      } catch (error) {
        reject(new Error('invalid_json'));
      }
    });
  });
}

function createApp() {
  return http.createServer(async (req, res) => {
    if (req.method === 'GET' && req.url === '/health') {
      writeJson(res, 200, { status: 'ok' });
      return;
    }

    if (req.method === 'GET' && req.url === '/api/dogs') {
      writeJson(res, 200, { dogs: mockDogs });
      return;
    }

    if (req.method === 'POST' && req.url === '/api/dogs/analyze') {
      try {
        const body = await parseBody(req);
        const name = body.name || 'Unknown';

        writeJson(res, 200, {
          dog: { name, breed: body.breed || 'Unknown' },
          analysis: 'Temporary mock analysis result',
          recommendation: 'Consult a veterinarian if symptoms continue.'
        });
      } catch (error) {
        writeJson(res, 400, { error: 'Request body must be valid JSON.' });
      }
      return;
    }

    writeJson(res, 404, { error: 'Not Found' });
  });
}

module.exports = {
  createApp
};
