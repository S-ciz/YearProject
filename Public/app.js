/* CHORDLAB frontend
   Set API_BASE_URL to your backend URL. Expected generation endpoint:
   POST /api/generate
   Body: { model: "Markov"|"LSTM"|"Transformer", key:"C:maj",
           chords:["C","Am","F","G"], num_chords:4, tempo:96 }
   Suggested response: { chords:["C","G","Am","F"], model:"LSTM" }
   Optional analytics endpoint: GET /api/analytics?model=Markov
   Expected analytics response: { epoch:100, train_loss:1.5548, test_loss:1.9177,
     accuracy:0.4171, precision:0.1417, f1:0.1168,
     train_loss_history:[...], test_loss_history:[...] }
*/
const API_BASE_URL = ""; // Example: "http://localhost:5000"
const GENERATE_ENDPOINT = "/api/generate";
const ANALYTICS_ENDPOINT = "/api/analytics";

const chordData = {
  C:{notes:["C4","E4","G4"],color:"#c7a7ff",quality:"MAJOR",degree:"I"},
  Dm:{notes:["D4","F4","A4"],color:"#72d8d0",quality:"MINOR",degree:"ii"},
  Em:{notes:["E4","G4","B4"],color:"#9da8ff",quality:"MINOR",degree:"iii"},
  F:{notes:["F4","A4","C5"],color:"#f4a6d8",quality:"MAJOR",degree:"IV"},
  G:{notes:["G4","B4","D5"],color:"#ffc38b",quality:"MAJOR",degree:"V"},
  Am:{notes:["A4","C5","E5"],color:"#8bd5a9",quality:"MINOR",degree:"vi"},
  Bdim:{notes:["B3","D4","F4"],color:"#e0a0a0",quality:"DIM",degree:"vii°"},
  Cmaj7:{notes:["C4","E4","G4","B4"],color:"#c7a7ff",quality:"MAJ7",degree:"Imaj7"},
  Dm7:{notes:["D4","F4","A4","C5"],color:"#72d8d0",quality:"MIN7",degree:"ii7"},
  Fmaj7:{notes:["F4","A4","C5","E5"],color:"#f4a6d8",quality:"MAJ7",degree:"IVmaj7"},
  G7:{notes:["G3","B3","D4","F4"],color:"#ffc38b",quality:"DOM7",degree:"V7"}
};
const initialProgression = ["C","Am","F","G"];
let progression = [...initialProgression];
let selectedModel = "Markov";
let selectedChordIndex = 0;
let audioStarted = false;
let synth = null;
let activeTimeouts = [];
let isPlaying = false;
let playStartedAt = 0;
let playTimer = null;
let toastTimer = null;
const $ = (selector, root=document) => root.querySelector(selector);
const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];

function showToast(message) {
  $("#toast-message").textContent = message;
  $("#toast").classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("#toast").classList.remove("show"), 2700);
}
function chordInfo(name) {
  return chordData[name] || chordData.C;
}
function chordDisplayName(name) {
  return name === "Bdim" ? "B°" : name;
}
function renderProgression() {
  const list = $("#progression-list");
  list.innerHTML = "";
  progression.forEach((chord, index) => {
    const info = chordInfo(chord);
    const card = document.createElement("div");
    card.className = "progression-chord" + (index === selectedChordIndex ? " active-chord" : "");
    card.style.setProperty("--chord-color", info.color);
    card.draggable = true;
    card.dataset.index = String(index);
    card.innerHTML = `<span class="pc-index">${String(index+1).padStart(2,"0")}</span><span class="pc-name">${chordDisplayName(chord)}</span><small>${info.quality}</small><button class="pc-remove" aria-label="Remove ${chord}" title="Remove chord">×</button>`;
    card.addEventListener("click", event => {
      if (event.target.closest(".pc-remove")) return;
      selectedChordIndex = index;
      renderProgression();
      playSingleChord(chord);
    });
    card.querySelector(".pc-remove").addEventListener("click", event => {
      event.stopPropagation();
      progression.splice(index, 1);
      selectedChordIndex = Math.max(0, Math.min(selectedChordIndex, progression.length-1));
      renderProgression();
    });
    card.addEventListener("dragstart", event => {
      event.dataTransfer.setData("text/chord-index", String(index));
      event.dataTransfer.effectAllowed = "move";
      card.classList.add("dragging");
    });
    card.addEventListener("dragend", () => card.classList.remove("dragging"));
    card.addEventListener("dragover", event => event.preventDefault());
    card.addEventListener("drop", event => {
      event.preventDefault();
      const from = Number(event.dataTransfer.getData("text/chord-index"));
      if (!Number.isNaN(from) && from >= 0 && from < progression.length && from !== index) {
        const [moved] = progression.splice(from,1);
        progression.splice(index,0,moved);
        selectedChordIndex = index;
        renderProgression();
      } else {
        const newChord = event.dataTransfer.getData("text/chord");
        if (chordData[newChord]) { progression.splice(index,0,newChord); renderProgression(); }
      }
    });
    list.appendChild(card);
  });
  $("#drop-hint").hidden = progression.length > 0;
  $("#progress-count").textContent = `${progression.length} CHORD${progression.length === 1 ? "" : "S"}`;
  $("#player-subtitle").innerHTML = `${progression.map(chordDisplayName).join(" · ") || "No chords yet"} <span>•</span> C Major`;
  $("#player-title").textContent = progression.length ? "Your progression" : "Empty progression";
  $("#progression-note").textContent = progression.length >= 4 ? "A classic, balanced progression" : progression.length ? "Your idea is taking shape" : "Add chords to get started";
  renderPiano();
}
function addChord(chord, atIndex=progression.length) {
  if (!chordData[chord]) return;
  const max = Number($("#chord-count").value);
  if (progression.length >= 10) { showToast("Progression limit reached (10 chords)"); return; }
  progression.splice(Math.max(0, Math.min(atIndex, progression.length)), 0, chord);
  selectedChordIndex = Math.min(atIndex, progression.length-1);
  renderProgression();
}
function renderPiano() {
  const piano = $("#piano");
  piano.innerHTML = "";
  const whites = ["C","D","E","F","G","A","B"];
  const blacks = [{note:"C#",left:10.7},{note:"D#",left:24.9},{note:"F#",left:53.4},{note:"G#",left:67.7},{note:"A#",left:81.9}];
  whites.forEach(note => {
    const key = document.createElement("div");
    key.className = "piano-key";
    key.dataset.note = `${note}4`;
    key.title = `${note}4`;
    key.addEventListener("click", () => playNotes([`${note}4`]));
    piano.appendChild(key);
  });
  blacks.forEach(item => {
    const key = document.createElement("div");
    key.className = "piano-key black";
    key.style.left = `${item.left}%`;
    key.dataset.note = `${item.note}4`;
    key.title = `${item.note}4`;
    key.addEventListener("click", () => playNotes([`${item.note}4`]));
    piano.appendChild(key);
  });
}
function highlightNotes(notes, duration=420) {
  $$(".piano-key").forEach(key => key.classList.toggle("active", notes.includes(key.dataset.note)));
  setTimeout(() => $$(".piano-key").forEach(key => key.classList.remove("active")), duration);
}
async function ensureAudio() {
  if (typeof Tone === "undefined") {
    showToast("Audio library unavailable. Check your internet connection.");
    return false;
  }
  try {
    await Tone.start();
    if (!synth) synth = new Tone.PolySynth(Tone.Synth, { oscillator:{type:"triangle"}, envelope:{attack:0.025, decay:0.25, sustain:0.35, release:1.1} }).toDestination();
    audioStarted = true;
    return true;
  } catch (error) {
    showToast("Audio could not start. Click play again to allow audio.");
    return false;
  }
}
async function playNotes(notes, duration=0.8) {
  if (!await ensureAudio()) return;
  const vol = Number($("#volume").value) / 100;
  synth.volume.value = vol === 0 ? -60 : -20 + 20 * Math.log10(vol);
  synth.triggerAttackRelease(notes, duration);
  highlightNotes(notes.map(n => n.replace(/([A-G]#?)\d/, "$1") + "4"));
}
async function playSingleChord(chord) {
  const info = chordInfo(chord);
  await playNotes(info.notes, 1.15);
  $("#playing-label").textContent = `PREVIEW · ${chordDisplayName(chord).toUpperCase()}`;
}
async function playProgression() {
  if (!progression.length) { showToast("Add at least one chord first"); return; }
  if (isPlaying) { stopPlayback(); return; }
  if (!await ensureAudio()) return;
  isPlaying = true;
  $("#play-icon").textContent = "Ⅱ";
  $("#playing-label").textContent = "PLAYING PROGRESSION";
  playStartedAt = Date.now();
  const beatMs = 60000 / Number($("#tempo").value);
  let index = 0;
  const playNext = () => {
    if (!isPlaying || index >= progression.length) {
      if (isPlaying) { stopPlayback(); $("#playing-label").textContent = "PLAYBACK COMPLETE"; $("#time-label").textContent = "00:00"; $("#play-progress").style.width = "100%"; }
      return;
    }
    selectedChordIndex = index;
    renderProgression();
    const chord = progression[index];
    const notes = chordInfo(chord).notes;
    const vol = Number($("#volume").value) / 100;
    synth.volume.value = vol === 0 ? -60 : -20 + 20 * Math.log10(vol);
    synth.triggerAttackRelease(notes, Math.max(.5, beatMs * 3.6 / 1000));
    highlightNotes(notes, beatMs * 3.5);
    $("#time-label").textContent = `00:${String(Math.round(index * beatMs * 4 / 1000)).padStart(2,"0")}`;
    $("#play-progress").style.width = `${((index+1)/progression.length)*100}%`;
    index++;
    activeTimeouts.push(setTimeout(playNext, beatMs * 4));
  };
  playNext();
}
function stopPlayback() {
  isPlaying = false;
  activeTimeouts.forEach(clearTimeout);
  activeTimeouts = [];
  $("#play-icon").textContent = "▶";
  $("#playing-label").textContent = "READY TO PLAY";
  if (playTimer) clearInterval(playTimer);
}
function setupDragAndDrop() {
  $$("#chord-palette [draggable=true], .extra-chords [draggable=true]").forEach(button => {
    button.addEventListener("dragstart", event => {
      event.dataTransfer.setData("text/chord", button.dataset.chord);
      event.dataTransfer.effectAllowed = "copy";
      button.classList.add("dragging");
    });
    button.addEventListener("dragend", () => button.classList.remove("dragging"));
    button.addEventListener("click", () => addChord(button.dataset.chord));
  });
  const zone = $("#drop-zone");
  zone.addEventListener("dragover", event => { event.preventDefault(); zone.classList.add("drag-over"); event.dataTransfer.dropEffect = "copy"; });
  zone.addEventListener("dragleave", event => { if (!zone.contains(event.relatedTarget)) zone.classList.remove("drag-over"); });
  zone.addEventListener("drop", event => {
    event.preventDefault();
    zone.classList.remove("drag-over");
    const chord = event.dataTransfer.getData("text/chord");
    const index = event.dataTransfer.getData("text/chord-index");
    if (chordData[chord]) addChord(chord);
    else if (index !== "" && !Number.isNaN(Number(index))) {
      const from = Number(index);
      if (from >= 0 && from < progression.length) {
        const [moved] = progression.splice(from,1);
        progression.push(moved);
        renderProgression();
      }
    }
  });
}
function chooseModel(model) {
  selectedModel = model;
  $$(".model-option").forEach(button => button.classList.toggle("selected", button.dataset.model === model));
  const descriptions = {
    Markov:["Markov model","Learns transitions between chord tokens from your training data."],
    LSTM:["LSTM sequence model","Uses recurrent memory to learn patterns across chord sequences."],
    Transformer:["Transformer model","Uses attention to capture relationships across the chord sequence."]
  };
  $("#model-info-title").textContent = descriptions[model][0];
  $("#model-info-text").textContent = descriptions[model][1];
}
function makeDemoProgression(count, seed) {
  const pools = {
    Markov:["C","Am","F","G","Dm","Em","G7","Cmaj7","Fmaj7","Dm7"],
    LSTM:["Am","F","C","G","Dm","Em","Fmaj7","Cmaj7","Dm7","G7"],
    Transformer:["Cmaj7","Dm7","Em","Fmaj7","G7","Am","Dm","C","F","G"]
  };
  const pool = pools[seed] || pools.Markov;
  let current = progression.length ? progression[progression.length-1] : "C";
  const result = [];
  for (let i=0;i<count;i++) {
    let candidates = pool.filter(ch => ch !== current);
    if (i === 0 && progression.length) candidates = pool.filter(ch => ch === current || chordInfo(ch));
    const next = candidates[Math.floor(Math.random()*candidates.length)];
    result.push(next);
    current = next;
  }
  return result;
}
async function generateProgression() {
  const button = $("#generate-button");
  const count = Number($("#chord-count").value);
  const requestBody = {model:selectedModel,key:"C:maj",chords:[...progression],num_chords:count,tempo:Number($("#tempo").value)};
  button.disabled = true;
  button.querySelector("span:nth-child(2)").textContent = "Generating...";
  button.classList.add("loading");
  try {
    let generated;
    if (API_BASE_URL) {
      const response = await fetch(`${API_BASE_URL}${GENERATE_ENDPOINT}`, {
        method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(requestBody)
      });
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      const data = await response.json();
      generated = data.chords || data.progression || data.generated_chords;
      if (!Array.isArray(generated) || !generated.length) throw new Error("API response must include a non-empty chords array");
      generated = generated.filter(ch => chordData[ch]);
      if (!generated.length) throw new Error("The API returned chords not configured in this UI");
      $("#api-mode-label").textContent = "API connected";
      alert("api connected")
    } else {
      await new Promise(resolve => setTimeout(resolve, 650));
      generated = makeDemoProgression(count, selectedModel);
      $("#api-mode-label").textContent = "Demo mode";
    }
    progression = generated.slice(0,count);
    selectedChordIndex = 0;
    renderProgression();
    showToast(`${selectedModel} generated ${progression.length} chords`);
  } catch (error) {
    console.error(error);
    showToast(`Generation failed: ${error.message}`);
  } finally {
    button.disabled = false;
    button.querySelector("span:nth-child(2)").textContent = "Generate progression";
    button.classList.remove("loading");
  }
}
function switchPage(page) {
  $$(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.page === page));
  $$(".page").forEach(section => section.classList.remove("active-page"));
  $(`#${page}-page`).classList.add("active-page");
  $("#breadcrumb-title").textContent = ({studio:"Studio",analytics:"Analytics",models:"Model lab"})[page];
  if (page === "analytics") loadAnalytics();
  window.scrollTo({top:0,behavior:"smooth"});
}
function buildHistory(finalValue, startValue, curvature, noise=0.05) {
  const values = [];
  for (let i=0;i<100;i++) {
    const t=i/99;
    const value=startValue+(finalValue-startValue)*(1-Math.exp(-curvature*t))/(1-Math.exp(-curvature));
    values.push(Math.max(.15,value+(Math.random()-.5)*noise*(1-t*.4)));
  }
  values[99]=finalValue;
  return values;
}
const defaultMetrics = {epoch:100,train_loss:1.5548,test_loss:1.9177,accuracy:.4171,precision:.1417,f1:.1168};
function pathFor(values, min=0.8, max=3.0) {
  return values.map((value,i) => `${i===0?"M":"L"} ${(i/(values.length-1))*600} ${202-((value-min)/(max-min))*192}`).join(" ");
}
function drawLossChart(train, test) {
  const trainPath=pathFor(train), testPath=pathFor(test);
  $("#train-line").setAttribute("d",trainPath);
  $("#test-line").setAttribute("d",testPath);
  $("#train-area").setAttribute("d",`${trainPath} L 600 220 L 0 220 Z`);
}
function drawQuality(metrics) {
  const items=[["Accuracy",metrics.accuracy,"#a58bff"],["Precision",metrics.precision,"#4de0c1"],["F1 score",metrics.f1,"#ffb875"]];
  $("#quality-chart").innerHTML=items.map(([name,value,color])=>`<div class="quality-row"><span class="quality-name">${name}</span><div class="quality-track"><div class="quality-fill" style="width:${Math.max(0,Math.min(100,value*100))}%;background:${color}"></div></div><span class="quality-value">${Number(value).toFixed(4)}</span></div>`).join("");
}
function updateMetrics(metrics, trainHistory, testHistory, model) {
  $("#metric-accuracy").innerHTML=`${(metrics.accuracy*100).toFixed(2)}<span>%</span>`;
  $("#metric-train-loss").textContent=Number(metrics.train_loss).toFixed(4);
  $("#metric-test-loss").textContent=Number(metrics.test_loss).toFixed(4);
  $("#metric-f1").textContent=Number(metrics.f1).toFixed(4);
  $("#loss-gap").textContent=(Number(metrics.test_loss)-Number(metrics.train_loss)).toFixed(4);
  $("#f1-meter span").style.width=`${Math.max(0,Math.min(100,metrics.f1*100))}%`;
  drawLossChart(trainHistory,testHistory);
  drawQuality(metrics);
  $("#metrics-table-body").innerHTML=`<tr><td><span class="table-model"><i class="table-dot"></i>${model}</span></td><td>${metrics.epoch||100} / ${metrics.epoch||100}</td><td>${Number(metrics.train_loss).toFixed(4)}</td><td>${Number(metrics.test_loss).toFixed(4)}</td><td><span class="table-accent">${Number(metrics.accuracy).toFixed(4)}</span></td><td>${Number(metrics.precision).toFixed(4)}</td><td>${Number(metrics.f1).toFixed(4)}</td></tr>`;
}
async function loadAnalytics() {
  const model=$("#analytics-model").value || selectedModel;
  let metrics={...defaultMetrics}, train=buildHistory(1.5548,2.9,3.1,.2), test=buildHistory(1.9177,2.95,2.0,.18);
  if (API_BASE_URL) {
    try {
      const response=await fetch(`${API_BASE_URL}${ANALYTICS_ENDPOINT}?model=${encodeURIComponent(model)}`);
      if (!response.ok) throw new Error(`Analytics API returned ${response.status}`);
      const data=await response.json();
      metrics={...metrics,...data};
      if (Array.isArray(data.train_loss_history) && data.train_loss_history.length>1) train=data.train_loss_history;
      if (Array.isArray(data.test_loss_history) && data.test_loss_history.length>1) test=data.test_loss_history;
    } catch(error) { console.warn(error); showToast("Could not load live analytics; showing sample metrics"); }
  }
  updateMetrics(metrics,train,test,model);
  $$("#train-bars, #test-bars").forEach((el,idx)=>{
    el.innerHTML="";
    const vals=idx===0?train:test;
    for(let i=0;i<18;i++){const bar=document.createElement("span");bar.style.height=`${10+Math.min(25,vals[Math.floor(i/17*(vals.length-1))]*8)}px`;el.appendChild(bar);}
  });
}
function downloadJson(filename, data) {
  const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");a.href=url;a.download=filename;a.click();URL.revokeObjectURL(url);
}
function init() {
  renderProgression();
  setupDragAndDrop();
  $$(".nav-item").forEach(button=>button.addEventListener("click",()=>switchPage(button.dataset.page)));
  $$(".model-option").forEach(button=>button.addEventListener("click",()=>chooseModel(button.dataset.model)));
  $$(".use-model").forEach(button=>button.addEventListener("click",()=>{chooseModel(button.dataset.model);switchPage("studio");showToast(`${button.dataset.model} selected`);}));
  $("#generate-button").addEventListener("click",generateProgression);
  $("#clear-progression").addEventListener("click",()=>{progression=[];selectedChordIndex=0;stopPlayback();renderProgression();});
  $("#shuffle-progression").addEventListener("click",()=>{if(!progression.length){showToast("Add chords before shuffling");return;}progression.sort(()=>Math.random()-.5);renderProgression();showToast("Progression shuffled");});
  $("#tempo").addEventListener("input",event=>$("#tempo-value").textContent=event.target.value);
  $("#play-button").addEventListener("click",playProgression);
  $("#prev-chord").addEventListener("click",()=>{if(!progression.length)return;selectedChordIndex=(selectedChordIndex-1+progression.length)%progression.length;renderProgression();playSingleChord(progression[selectedChordIndex]);});
  $("#next-chord").addEventListener("click",()=>{if(!progression.length)return;selectedChordIndex=(selectedChordIndex+1)%progression.length;renderProgression();playSingleChord(progression[selectedChordIndex]);});
  $("#analytics-model").addEventListener("change",loadAnalytics);
  $("#export-analytics").addEventListener("click",()=>downloadJson("chordlab-analytics.json",{model:$("#analytics-model").value,epoch:100,...defaultMetrics}));
  $("#download-midi").addEventListener("click",()=>downloadJson("chordlab-progression.json",{key:"C:maj",tempo:Number($("#tempo").value),model:selectedModel,chords:progression}));
  loadAnalytics();
}
document.addEventListener("DOMContentLoaded",init);
