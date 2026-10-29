
/*
 * ChordLab
 * Frontend for an RNN-based next-chord predictor.
 *
 * Expected API:
 * POST http://127.0.0.1:5000/generate
 * {
 *   "chords": ["C:maj", "F:maj", "G:maj"],
 *   "num_chords": 4
 * }
 *
 * Expected response:
 * {
 *   "generated_chords": ["A:min", "F:maj", "G:maj", "C:maj"]
 * }
 *
 * Adapt API_URL and the request/response fields to match
 * your actual Flask or Node.js backend.
 */

const API_URL = "http://127.0.0.1:5000/generate";

const ROOTS = [
    "A#", "A", "B", "C#", "C", "D#",
    "D", "E", "F#", "F", "G#", "G"
];

const TYPES = ["aug", "dim", "dom7", "maj", "min", "sus"];

const TYPE_NAMES = {
    aug: "Augmented",
    dim: "Diminished",
    dom7: "Dominant 7th",
    maj: "Major",
    min: "Minor",
    sus: "Suspended"
};

const CHORDS = ROOTS.flatMap(root =>
    TYPES.map(type => `${root}:${type}`)
);

const VALID_COUNTS = [2, 4, 6, 8, 10];

let sequence = [];
let generatedChords = [];
let history = [];
let activeFilter = "all";
let draggedSequenceIndex = null;
let audioContext = null;
let playbackToken = 0;
let isPlaying = false;

const $ = id => document.getElementById(id);

const chordLibrary = $("chordLibrary");
const timeline = $("timeline");
const emptyState = $("emptyState");
const searchInput = $("searchInput");
const sequenceCount = $("sequenceCount");
const visibleCount = $("visibleCount");
const chordCount = $("chordCount");
const generateButton = $("generateButton");
const generationStatus = $("generationStatus");
const resultsSection = $("resultsSection");
const resultTimeline = $("resultTimeline");
const modelStatus = $("modelStatus");

chordCount.textContent = `${CHORDS.length} chords`;

// --------------------------------------------------
// CHORD LIBRARY
// --------------------------------------------------

function getChordType(chord) {
    return chord.split(":")[1];
}

function renderLibrary() {
    const search = searchInput.value.trim().toLowerCase();

    const filtered = CHORDS.filter(chord => {
        const matchesSearch = chord.toLowerCase().includes(search);

        const matchesType =
            activeFilter === "all" ||
            getChordType(chord) === activeFilter;

        return matchesSearch && matchesType;
    });

    visibleCount.textContent = filtered.length;

    chordLibrary.innerHTML = filtered.map(chord => `
        <button
            class="chord-card"
            draggable="true"
            data-chord="${chord}"
            aria-label="Add ${chord} to progression"
            title="Drag or click to add ${chord}"
        >
            <span class="chord-name">${chord}</span>
            <span class="chord-type">${TYPE_NAMES[getChordType(chord)]}</span>
        </button>
    `).join("");

    chordLibrary.querySelectorAll(".chord-card").forEach(card => {
        card.addEventListener("click", () => {
            addChord(card.dataset.chord);
        });

        card.addEventListener("dragstart", event => {
            event.dataTransfer.setData(
                "text/plain",
                card.dataset.chord
            );

            event.dataTransfer.effectAllowed = "copy";
        });
    });
}

searchInput.addEventListener("input", renderLibrary);

$("typeFilters").addEventListener("click", event => {
    const button = event.target.closest("[data-type]");

    if (!button) return;

    activeFilter = button.dataset.type;

    document.querySelectorAll(".filter").forEach(filter => {
        filter.classList.toggle("active", filter === button);
    });

    renderLibrary();
});

// --------------------------------------------------
// SEQUENCE EDITOR
// --------------------------------------------------

function saveHistory() {
    history.push([...sequence]);

    // Keep a manageable undo history.
    if (history.length > 50) {
        history.shift();
    }

    $("undoButton").disabled = history.length === 0;
}

function addChord(chord, index = sequence.length) {
    if (!CHORDS.includes(chord)) return;

    saveHistory();

    sequence.splice(index, 0, chord);

    clearResults();
    renderSequence();

    setStatus(`${chord} added to your progression.`);
}

function removeChord(index) {
    if (index < 0 || index >= sequence.length) return;

    saveHistory();

    sequence.splice(index, 1);

    clearResults();
    renderSequence();

    setStatus("Chord removed.");
}

function moveChord(fromIndex, toIndex) {
    if (
        fromIndex === null ||
        fromIndex === toIndex ||
        fromIndex < 0 ||
        fromIndex >= sequence.length ||
        toIndex < 0 ||
        toIndex >= sequence.length
    ) {
        return;
    }

    saveHistory();

    const [chord] = sequence.splice(fromIndex, 1);

    sequence.splice(toIndex, 0, chord);

    clearResults();
    renderSequence();
}

function renderSequence() {
    sequenceCount.textContent =
        `${sequence.length} chord${sequence.length === 1 ? "" : "s"}`;

    emptyState.hidden = sequence.length > 0;

    timeline.querySelectorAll(".sequence-chip").forEach(chip => {
        chip.remove();
    });

    sequence.forEach((chord, index) => {
        const chip = document.createElement("div");

        chip.className = "sequence-chip";
        chip.draggable = true;
        chip.dataset.index = index;

        chip.innerHTML = `
            <span class="position">${String(index + 1).padStart(2, "0")}</span>
            <span class="chord-name"></span>
            <button class="remove-chip" aria-label="Remove chord"
                    title="Remove chord">×</button>
        `;

        chip.querySelector(".chord-name").textContent = chord;

        chip.querySelector(".remove-chip").addEventListener("click", event => {
            event.stopPropagation();
            removeChord(index);
        });

        chip.addEventListener("dragstart", event => {
            draggedSequenceIndex = index;

            event.dataTransfer.setData(
                "application/x-chordlab-index",
                String(index)
            );

            event.dataTransfer.effectAllowed = "move";
        });

        chip.addEventListener("dragover", event => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "move";
        });

        chip.addEventListener("drop", event => {
            event.preventDefault();

            const fromIndex = Number(
                event.dataTransfer.getData(
                    "application/x-chordlab-index"
                )
            );

            if (Number.isInteger(fromIndex)) {
                moveChord(fromIndex, index);
            }
        });

        chip.addEventListener("dragend", () => {
            draggedSequenceIndex = null;
        });

        timeline.appendChild(chip);
    });

    $("playInputButton").disabled = sequence.length === 0;
    generateButton.disabled = sequence.length === 0;
}

timeline.addEventListener("dragover", event => {
    event.preventDefault();
    timeline.classList.add("drag-over");
});

timeline.addEventListener("dragleave", event => {
    if (!timeline.contains(event.relatedTarget)) {
        timeline.classList.remove("drag-over");
    }
});

timeline.addEventListener("drop", event => {
    event.preventDefault();

    timeline.classList.remove("drag-over");

    const chord = event.dataTransfer.getData("text/plain");
    const indexData = event.dataTransfer.getData(
        "application/x-chordlab-index"
    );

    // Moving an existing chord within the timeline.
    if (indexData !== "") {
        const fromIndex = Number(indexData);
        const target = event.target.closest(".sequence-chip");

        if (target) {
            moveChord(fromIndex, Number(target.dataset.index));
        }

        return;
    }

    // Adding a new chord from the library.
    if (CHORDS.includes(chord)) {
        const target = event.target.closest(".sequence-chip");

        const insertAt = target
            ? Number(target.dataset.index)
            : sequence.length;

        addChord(chord, insertAt);
    }
});

$("undoButton").addEventListener("click", () => {
    if (history.length === 0) return;

    sequence = history.pop();

    clearResults();
    renderSequence();

    $("undoButton").disabled = history.length === 0;

    setStatus("Previous progression restored.");
});

$("clearButton").addEventListener("click", () => {
    if (sequence.length === 0) return;

    saveHistory();

    sequence = [];

    clearResults();
    renderSequence();

    setStatus("Progression cleared.");
});

// --------------------------------------------------
// RNN GENERATION
// --------------------------------------------------

function setStatus(message, type = "") {
    generationStatus.textContent = message;
    generationStatus.className = "generation-status";

    if (type) {
        generationStatus.classList.add(type);
    }
}

function clearResults() {
    generatedChords = [];
    resultsSection.hidden = true;
    resultTimeline.innerHTML = "";
}

function renderResults() {
    resultsSection.hidden = false;
    resultTimeline.innerHTML = "";

    generatedChords.forEach((chord, index) => {
        const chip = document.createElement("div");

        chip.className = "sequence-chip generated";

        chip.innerHTML = `
            <span class="position">${String(index + 1).padStart(2, "0")}</span>
            <span class="chord-name"></span>
        `;

        chip.querySelector(".chord-name").textContent = chord;

        resultTimeline.appendChild(chip);
    });
}

generateButton.addEventListener("click", generateContinuation);

async function generateContinuation() {
    if (sequence.length === 0) {
        setStatus("Add at least one input chord first.", "error");
        return;
    }

    const requestedCount = Number($("generateCount").value);

    if (!VALID_COUNTS.includes(requestedCount)) {
        setStatus("Select a valid number of chords.", "error");
        return;
    }

    const inputSequence = [...sequence];

    generateButton.disabled = true;
    generateButton.textContent = "Generating...";
    modelStatus.textContent = "CONNECTING TO RNN";

    setStatus(
        `Sending ${inputSequence.length} input chords to the RNN...`
    );

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                chords: inputSequence,
                num_chords: requestedCount,
                key: "C:maj"
            })
        });

        if (!response.ok) {
            throw new Error(
                `The model API returned HTTP ${response.status}.`
            );
        }

        const data = await response.json();

        // Accept a few common response formats.
        const predictions =
            data.generated_chords ??
            data.continuation ??
            data.chords;

        if (!Array.isArray(predictions)) {
            throw new Error(
                "Invalid API response: expected an array of generated chords."
            );
        }

        if (predictions.length !== requestedCount) {
            throw new Error(
                `The model returned ${predictions.length} chords; ` +
                `${requestedCount} were requested.`
            );
        }

        const invalidChords = predictions.filter(
            chord => !CHORDS.includes(chord)
        );

        if (invalidChords.length > 0) {
            throw new Error(
                `The model returned unsupported chords: ${invalidChords.join(", ")}`
            );
        }

        // Prevent a stale request from overwriting a changed progression.
        if (
            inputSequence.length !== sequence.length ||
            inputSequence.some((chord, i) => chord !== sequence[i])
        ) {
            setStatus(
                "Your input changed while the model was working. Generate again.",
                "error"
            );
            return;
        }

        generatedChords = predictions;

        renderResults();

        modelStatus.textContent = "RNN CONNECTED";

        setStatus(
            `Successfully generated ${generatedChords.length} new chords.`,
            "success"
        );

    } catch (error) {
        console.error("RNN generation failed:", error);

        modelStatus.textContent = "RNN DISCONNECTED";

        setStatus(
            `${error.message} Check that your API is running at ${API_URL}.`,
            "error"
        );
    } finally {
        generateButton.disabled = sequence.length === 0;
        generateButton.innerHTML = "<span>✦</span> Generate continuation";
    }
}

$("discardButton").addEventListener("click", () => {
    clearResults();
    setStatus("Generated results cleared.");
});

$("keepButton").addEventListener("click", () => {
    if (generatedChords.length === 0) return;

    saveHistory();

    sequence = [...sequence, ...generatedChords];

    clearResults();
    renderSequence();

    setStatus("Generated chords added to your progression.", "success");
});

$("playFullButton").addEventListener("click", () => {
    if (generatedChords.length === 0) return;

    playSequence([...sequence, ...generatedChords], "Full progression");
});

// --------------------------------------------------
// WEB AUDIO PLAYER
// --------------------------------------------------

// MIDI pitch classes relative to C.
const ROOT_SEMITONES = {
    C: 0,
    "C#": 1,
    D: 2,
    "D#": 3,
    E: 4,
    F: 5,
    "F#": 6,
    G: 7,
    "G#": 8,
    A: 9,
    "A#": 10,
    B: 11
};

const CHORD_INTERVALS = {
    maj: [0, 4, 7],
    min: [0, 3, 7],
    dim: [0, 3, 6],
    aug: [0, 4, 8],
    dom7: [0, 4, 7, 10],
    sus: [0, 5, 7]
};

function midiToFrequency(midi) {
    return 440 * Math.pow(2, (midi - 69) / 12);
}

function chordFrequencies(chord) {
    const [root, type] = chord.split(":");

    const rootSemitone = ROOT_SEMITONES[root];
    const intervals = CHORD_INTERVALS[type];

    if (rootSemitone === undefined || !intervals) {
        throw new Error(`Unsupported chord: ${chord}`);
    }

    // C4 is MIDI 60. Put all roots in a comfortable register.
    const rootMidi = 60 + rootSemitone;

    return intervals.map(interval =>
        midiToFrequency(rootMidi + interval)
    );
}

async function ensureAudioContext() {
    const AudioContextClass =
        window.AudioContext || window.webkitAudioContext;

    if (!AudioContextClass) {
        throw new Error("Your browser does not support Web Audio.");
    }

    if (!audioContext) {
        audioContext = new AudioContextClass();
    }

    if (audioContext.state === "suspended") {
        await audioContext.resume();
    }

    return audioContext;
}

function playChordAt(chord, startTime, duration) {
    const frequencies = chordFrequencies(chord);

    frequencies.forEach((frequency, index) => {
        const oscillator = audioContext.createOscillator();
        const gain = audioContext.createGain();

        oscillator.type = "triangle";
        oscillator.frequency.value = frequency;

        // Keep multiple simultaneous notes from clipping.
        const volume = 0.12 / frequencies.length;

        gain.gain.setValueAtTime(0.0001, startTime);
        gain.gain.exponentialRampToValueAtTime(
            volume,
            startTime + 0.035
        );

        gain.gain.setValueAtTime(
            volume,
            startTime + duration * 0.72
        );

        gain.gain.exponentialRampToValueAtTime(
            0.0001,
            startTime + duration
        );

        oscillator.connect(gain);
        gain.connect(audioContext.destination);

        oscillator.start(startTime);
        oscillator.stop(startTime + duration + 0.02);
    });
}

async function playSequence(chords, title) {
    if (chords.length === 0) return;

    try {
        await ensureAudioContext();

        // Stop any previous scheduled sequence.
        playbackToken++;
        const currentToken = playbackToken;

        isPlaying = true;

        $("playerTitle").textContent = title;
        $("playerSubtitle").textContent =
            `Playing 0 / ${chords.length} chords`;

        const chordDuration = 0.85;
        const gap = 0.08;
        const startTime = audioContext.currentTime + 0.08;

        chords.forEach((chord, index) => {
            const time = startTime + index * (chordDuration + gap);

            playChordAt(chord, time, chordDuration);

            window.setTimeout(() => {
                if (currentToken !== playbackToken) return;

                $("playerSubtitle").textContent =
                    `Playing ${index + 1} / ${chords.length}: ${chord}`;
            }, Math.max(0, (time - audioContext.currentTime) * 1000));
        });

        const totalDuration =
            chords.length * (chordDuration + gap);

        window.setTimeout(() => {
            if (currentToken !== playbackToken) return;

            isPlaying = false;

            $("playerSubtitle").textContent =
                `Finished · ${chords.length} chords`;

        }, totalDuration * 1000);

    } catch (error) {
        console.error("Playback failed:", error);

        $("playerSubtitle").textContent = error.message;
    }
}

$("playInputButton").addEventListener("click", () => {
    playSequence(sequence, "Input progression");
});

$("stopButton").addEventListener("click", () => {
    playbackToken++;
    isPlaying = false;

    if (audioContext && audioContext.state === "running") {
        // Fade out currently sounding notes quickly.
        audioContext.suspend().then(() => {
            audioContext.resume();
        });
    }

    $("playerTitle").textContent = "ChordLab Player";
    $("playerSubtitle").textContent = "Playback stopped";
});

// --------------------------------------------------
// KEYBOARD SHORTCUTS AND INITIALIZATION
// --------------------------------------------------

document.addEventListener("keydown", event => {
    const target = event.target;

    if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
    ) {
        return;
    }

    if (event.key === "/" && !event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        searchInput.focus();
    }

    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();

        if (history.length > 0) {
            $("undoButton").click();
        }
    }
});

function initializeApp() {
    renderLibrary();
    renderSequence();

    $("undoButton").disabled = true;
    $("playInputButton").disabled = true;
    generateButton.disabled = true;

    setStatus("Add at least one chord to begin.");
}

initializeApp();
