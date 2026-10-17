const $ = (selector) => document.querySelector(selector);
const form = $('#estimate-form');
const categorySelect = $('#category');
const locationSelect = $('#location');
const roomSelect = $('#rooms');
const submit = $('#submit-button');

const money = (value) => new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value) + ' AZN';

function addOptions(select, values, selected, labels = {}) {
  select.replaceChildren(...values.map((value) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = labels[value] || value;
    if (value === selected) option.selected = true;
    return option;
  }));
}

async function initialize() {
  try {
    const [healthResponse, metadataResponse] = await Promise.all([
      fetch('/api/health'), fetch('/api/metadata')
    ]);
    if (!healthResponse.ok || !metadataResponse.ok) throw new Error('The server did not respond.');
    const health = await healthResponse.json();
    const metadata = await metadataResponse.json();
    if (!health.ready) throw new Error('The model is not ready.');
    addOptions(categorySelect, metadata.categories, metadata.categories[0], {
      'Yeni tikili': 'New-build apartment',
      'Köhnə tikili': 'Existing apartment',
      'Həyət evi/Bağ evi': 'House / garden home'
    });
    addOptions(locationSelect, metadata.locations, 'Nəsimi r.');
    addOptions(roomSelect, Array.from({ length: 30 }, (_, i) => String(i + 1)), '3');
    $('#stat-rows').textContent = Number(metadata.training_rows).toLocaleString('en-US');
    $('#hero-rows').textContent = Number(metadata.training_rows).toLocaleString('en-US');
    $('#stat-threshold').textContent = money(metadata.threshold_azn);
    $('#model-state').textContent = 'Ready';
    $('#footer-status').textContent = 'Estimator ready';
    $('#model-state').previousElementSibling.classList.add('ready');
    submit.disabled = false;
    categorySelect.dispatchEvent(new Event('change'));
  } catch (error) {
    $('#model-state').textContent = 'Offline';
    $('#footer-status').textContent = 'Server offline';
    submit.disabled = true;
    showError('Start the app with `python -m src.web_demo`, then refresh this page.');
  }
}

function showError(message) {
  const target = $('#result-error');
  target.textContent = message;
  target.classList.remove('hidden');
  $('#result-placeholder').classList.add('hidden');
  $('#result-output').classList.add('hidden');
}

function makeRequest() {
  const data = new FormData(form);
  const getNumber = (name) => Number(data.get(name));
  const house = data.get('category') === 'Həyət evi/Bağ evi';
  return {
    category: data.get('category'), location: data.get('location'),
    area_m2: getNumber('area_m2'), rooms: getNumber('rooms'),
    floor: house ? null : getNumber('floor'),
    n_floors: house ? null : getNumber('n_floors'),
    land_m2: house ? (getNumber('land_m2') || 0) : 0,
    has_repair: data.has('has_repair'),
    has_bill_of_sale: data.has('has_bill_of_sale'),
    has_mortgage: data.has('has_mortgage')
  };
}

function showResult(result, request) {
  $('#result-error').classList.add('hidden');
  $('#result-placeholder').classList.add('hidden');
  $('#result-output').classList.remove('hidden');
  $('#price-value').textContent = money(result.price_azn);
  $('#sqm-value').textContent = money(result.price_per_m2_azn);
  $('#tier-value').textContent = result.tier;
  $('#tier-symbol').textContent = result.tier === 'Premium' ? '✦' : '◈';
  $('#tier-symbol').style.color = result.tier === 'Premium' ? '#7b8500' : '#10b981';
  const leafShare = Math.round(result.tier_probability * 100);
  $('#probability-value').textContent = `${leafShare}%`;
  $('#probability-bar').style.width = `${leafShare}%`;
  const floorText = request.floor ? `, floor ${request.floor}/${request.n_floors}` : '';
  const categoryLabels = {'Yeni tikili':'New-build apartment','Köhnə tikili':'Existing apartment','Həyət evi/Bağ evi':'House / garden home'};
  const options = [request.has_repair ? 'repaired' : 'repair not specified', request.has_bill_of_sale ? 'title deed' : 'deed not specified'];
  $('#estimate-summary').textContent = `${categoryLabels[request.category] || request.category} · ${request.location} · ${request.area_m2} m² · ${request.rooms} rooms${floorText}. ${options.join(', ')}.${result.location_coordinate_approximate ? ' Approximate area coordinates used.' : ''}`;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const request = makeRequest();
  submit.disabled = true;
  submit.innerHTML = '<span class="spinner"></span> Calculating';
  $('#result-placeholder').classList.remove('hidden');
  $('#result-output').classList.add('hidden');
  $('#result-error').classList.add('hidden');
  try {
    const response = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not calculate an estimate.');
    showResult(result, request);
  } catch (error) {
    showError(error.message);
  } finally {
    submit.disabled = false;
    submit.innerHTML = 'Estimate price <span aria-hidden="true">→</span>';
  }
});

categorySelect.addEventListener('change', () => {
  const house = categorySelect.value === 'Həyət evi/Bağ evi';
  $('#floor-fields').classList.toggle('hidden', house);
  $('#land-field').classList.toggle('hidden', !house);
});

initialize();
