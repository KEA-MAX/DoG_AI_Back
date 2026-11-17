const test = require('node:test');
const assert = require('node:assert/strict');
const { once } = require('node:events');
const { createApp } = require('../app');

function makeRequest({ port, method, path, body }) {
  const payload = body ? JSON.stringify(body) : null;
  const url = `http://127.0.0.1:${port}${path}`;

  return fetch(url, {
    method,
    headers: payload
      ? {
          'Content-Type': 'application/json'
        }
      : undefined,
    body: payload
  });
}

test('GET /health returns server status', async () => {
  const app = createApp();
  app.listen(0);
  await once(app, 'listening');

  const address = app.address();
  const response = await makeRequest({ port: address.port, method: 'GET', path: '/health' });
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.deepEqual(payload, { status: 'ok' });

  app.close();
});

test('GET /api/dogs returns mock dogs list', async () => {
  const app = createApp();
  app.listen(0);
  await once(app, 'listening');

  const address = app.address();
  const response = await makeRequest({ port: address.port, method: 'GET', path: '/api/dogs' });
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.equal(Array.isArray(payload.dogs), true);
  assert.equal(payload.dogs.length, 2);

  app.close();
});

test('POST /api/dogs/analyze validates JSON body', async () => {
  const app = createApp();
  app.listen(0);
  await once(app, 'listening');

  const address = app.address();
  const response = await makeRequest({
    port: address.port,
    method: 'POST',
    path: '/api/dogs/analyze',
    body: { name: 'Pochi', breed: 'Shiba Inu' }
  });
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.equal(payload.dog.name, 'Pochi');
  assert.equal(payload.analysis, 'Temporary mock analysis result');

  app.close();
});
