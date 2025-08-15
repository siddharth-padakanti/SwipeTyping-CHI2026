// study.js

// ==== Monkey‐patch logging so we only record in main study, but always track swipes ====
window.STUDY_LOGGING_ENABLED = false;
const _origLogGesture = window.logGesture;
const _origLogWords   = window.logWords;

window.logGesture = function(type, sx, sy, ex, ey, skey, ekey) {
  // always count swipes for practice‐check
  if (type === "swipe") studyState.swipeCount++;
  // only record full logs in main study
  if (STUDY_LOGGING_ENABLED) {
    _origLogGesture(type, sx, sy, ex, ey, skey, ekey);
  }
};

window.logWords = function() {
  if (STUDY_LOGGING_ENABLED) {
    _origLogWords();
  }
};


// ==== Study state and trial lists ====
const studyState = {
  participantID: null,
  phase: null,          // "swipe1", "swipe2", "main"
  trialIndex: 0,
  sentences: [],
  swipeCount: 0,
  // for swipe1 blocks:
  blockIndex: 0,        // 0..3 (4 blocks)
  withinBlockIndex: 0,  // 0..4 (5 trials per block)
  blockTrials: []       // current block's randomized 5 words
};
window.studyState = studyState;

/**
 * PRACTICE 1: 5 long words (from finetune_data.csv), used across 4 blocks.
 * Each block: same 5 words in random order => 5 trials; 4 blocks => 20 total trials.
 * Chosen from the CSV and long enough for swiping:
 * - providing
 * - choosing
 * - reception
 * - auctions
 * - administrative
 */
const SWIPE1_WORDS = [
  "providing",
  "choosing",
  "reception",
  "auctions",
  "administrative"
];

/**
 * PRACTICE 2: 5 sentences; each contains a word that shares a ≥4-letter suffix
 * with one of the SWIPE1 words. All similar words were also taken from the CSV:
 * - excluding (similar to providing)
 * - addressing (similar to choosing)
 * - restoration (similar to reception)
 * - infections (similar to auctions)
 * - collaborative (similar to administrative)
 */
const SWIPE2_SENTENCES = [
  "The policy is excluding several old options this quarter",
  "We are addressing the complex issue through weekly updates",
  "After the storm, the restoration effort took many months",
  "Hospitals reported several infections across rural areas",
  "Our collaborative team improved results in record time"
];

// Keep your existing MAIN set untouched
const MAIN_SENTENCES = [
  "test one",
  "three musketeers",
  "pretty stuff",
  // "i see a bright sun in sky",
  // "we learn to love and give hope",
  // "she reads a book to know more",
  // "they play games and make new friends",
  // "he writes words to share his mind",
  // "we find joy in each new day",
  // "i feel calm when music plays",
  // "you can grow and learn each time",
  // "they walk and talk with good friends",
  // "she helps others to grow and learn",
  // "you hold hope in your heart today",
  // "we want to make life better now",
  // "he takes time to watch the sky",
  // "they see beauty in small simple things",
  // "you make a difference when you care",
  // "i learn new things and use them",
  // "she gives love and gets love back",
  // "we share food and stories with family",
  // "he finds peace in music and art",
  // "they write words that touch the soul",
  // "you show a kind act every day",
  // "i play and laugh in bright sunshine",
  // "we use our mind to solve problems",
  // "he knows that each day brings change",
  // "she buys and sells things with care",
  // "we meet many new people",
  // "i try and try until I succeed",
  // "she runs to stay strong",
  // "we eat and drink to stay healthy",
  // "you rest and take time to grow"
];

// ==== Helpers ====
function showSection(id) {
  document.querySelectorAll("section").forEach(s => s.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}

async function registerParticipant(rawID) {
  try {
    let resp = await fetch("/typing/register", {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ id: rawID })
    });
    if (resp.ok) {
      let { id } = await resp.json();
      return id;
    }
  } catch(e) { /* ignore */ }
  return rawID;
}

function shuffleTrials(array) {
  const a = array.slice();            
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}


// ==== Init ====
document.getElementById("btn-start-study")
  .addEventListener("click", async () => {
    const input = document.getElementById("participant-id").value.trim();
    if (!input) return alert("Please enter your ID first.");
    const uniqueID = await registerParticipant(input);
    studyState.participantID = uniqueID;
    showSection("main-menu");
  });

// ==== Main Menu ====
document.getElementById("btn-swipe1")
  .addEventListener("click", () => startPhase("swipe1"));
document.getElementById("btn-swipe2")
  .addEventListener("click", () => startPhase("swipe2"));
document.getElementById("btn-main-study")
  .addEventListener("click", () => startPhase("main"));


// ==== Phase control ====
function startPhase(phase) {
  studyState.phase = phase;
  studyState.trialIndex = 0;

  if (phase === "swipe1") {
    // 4 blocks × 5 trials each
    studyState.blockIndex = 0;
    studyState.withinBlockIndex = 0;
    studyState.blockTrials = shuffleTrials(SWIPE1_WORDS);
    STUDY_LOGGING_ENABLED = false;
    showSection("task-section");
    loadSwipe1Trial();
    return;
  }

  switch (phase) {
    case "swipe2":
      studyState.sentences = shuffleTrials(SWIPE2_SENTENCES); // 5 trials
      STUDY_LOGGING_ENABLED = false;
      break;
    case "main":
      studyState.sentences = shuffleTrials(MAIN_SENTENCES);
      STUDY_LOGGING_ENABLED = true;
      break;
  }

  showSection("task-section");
  loadTrial_Generic();
}

function endPhase() {
  if (studyState.phase === "main") {
    showSection("completion-section");
  } else {
    showSection("main-menu");
  }
}


// ==== Trial loaders ====

// Swipe Practice 1 (blocks/words)
function loadSwipe1Trial() {
  const promptHeader = document.getElementById("prompt-header");
  const counter = document.getElementById("trial-counter");
  const promptBox = document.getElementById("prompt-box");

  promptHeader.textContent = "Word";
  const blockNum = studyState.blockIndex + 1;    // 1..4
  const trialNum = studyState.withinBlockIndex + 1; // 1..5

  counter.innerHTML =
    `Block: <span class="badge badge-block">${blockNum}</span> | ` +
    `Trial: <span class="badge badge-trial">${trialNum}</span>/5`;

  const word = studyState.blockTrials[studyState.withinBlockIndex];
  promptBox.textContent = word;

  // reset interaction state
  studyState.swipeCount = 0;
  clearForNewTrial();
}

// Generic loader used by swipe2 + main (sentences)
function loadTrial_Generic() {
  const total = studyState.sentences.length;
  const current = studyState.trialIndex + 1;

  const isSwipe2 = studyState.phase === "swipe2";
  document.getElementById("prompt-header").textContent = "Sentence";

  document.getElementById("trial-counter").textContent =
    `Trial ${current}/${total}  Please type the following sentence`;
  document.getElementById("prompt-box").textContent =
    studyState.sentences[studyState.trialIndex];

  studyState.swipeCount = 0;
  clearForNewTrial();
}

function clearForNewTrial() {
  clearInput();                 // clears formattedInput & currentWord
  typedWords = [];              // actually empty your sentence buffer
  currentTypedWord = "";        // clear last prediction
  display.innerHTML = "";       // clear the display box
  display.style.borderColor = "#ccc";
  predictionBar.replaceChildren(); // remove old buttons
}


// ==== Enter handling (submit) ====
document.addEventListener("keydown", ev => {
  if (ev.key === "Enter" && document.getElementById("task-section").classList.contains("active")) {
    ev.preventDefault();
    ev.stopImmediatePropagation();
    handleSubmitSentence();
  }
}, true);

function handleSubmitSentence() {
  const displayEl = document.getElementById("display");

  // Auto-confirm any in-flight word
  if (formattedInput.length > 0 || currentTypedWord) {
      const last = (currentTypedWord || formattedInput.join("")).toLowerCase().trim();
      if (last) {
        typedWords.push(last);
        typedWords.push(" ");
      }
      clearInput();
  }

  let rawSentence = typedWords.join("");
  rawSentence = rawSentence.replace(/ {2,}/g, " ");
  rawSentence = rawSentence.trim();

  displayEl.textContent = rawSentence;
  displayEl.innerHTML += '<span class="blinking-cursor">|</span>';

  // Evaluate depending on phase
  if (studyState.phase === "swipe1") {
    // Expected is a single word
    const expected = document.getElementById("prompt-box").textContent.trim().toLowerCase();
    const actual = rawSentence.toLowerCase();

    // swipe check: at least one swipe used to enter the single word
    if (studyState.swipeCount < 1) {
      displayEl.style.borderColor = "red";
      alert("Please use a swipe for the word before submitting.");
      return;
    }

    if (actual !== expected) {
      displayEl.style.borderColor = "red";
      return;
    }

    // correct → advance within block / advance block
    displayEl.style.borderColor = "green";
    setTimeout(() => {
      studyState.withinBlockIndex++;
      if (studyState.withinBlockIndex < 5) {
        loadSwipe1Trial();
      } else {
        // next block or end
        studyState.blockIndex++;
        if (studyState.blockIndex < 4) {
          studyState.withinBlockIndex = 0;
          studyState.blockTrials = shuffleTrials(SWIPE1_WORDS); // reshuffle same 5 for each block
          loadSwipe1Trial();
        } else {
          endPhase();
        }
      }
    }, 400);

    return;
  }

  // swipe2 / main
  const expected = studyState.sentences[studyState.trialIndex].trim().toLowerCase();
  const actual   = rawSentence.toLowerCase();
  const wordCount = expected.split(/\s+/).length;

  if (actual !== expected) {
    displayEl.style.borderColor = "red";
    return;
  }

  // swipe-practice check (applies to both swipe1 and swipe2)
  if ((studyState.phase === "swipe2") && studyState.swipeCount < wordCount) {
    displayEl.style.borderColor = "red";
    alert("Please use a swipe for each word before submitting.");
    return;
  }

  // Pass → advance
  displayEl.style.borderColor = "green";
  setTimeout(() => {
    studyState.trialIndex++;
    if (studyState.trialIndex < studyState.sentences.length) {
      loadTrial_Generic();
    } else {
      endPhase();
    }
  }, 500);
}


// ==== Completion buttons ====
document.getElementById("btn-download-gesture-logs")
  .addEventListener("click", () => {
    downloadGestureCSV();
  });
document.getElementById("btn-download-word-logs")
  .addEventListener("click", () => {
    downloadWordCSV();
  });
