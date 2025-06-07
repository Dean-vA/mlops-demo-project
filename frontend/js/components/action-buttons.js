// Action Buttons Component (js/components/action-buttons.js)
class ActionButtonsComponent {
    constructor(state, services) {
        this.state = state;
        this.services = services;
        this.container = document.getElementById('action-buttons-component');
    }

    async init() {
        this.render();
        this.setupEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="transcribe-buttons">
                <button class="transcribe-btn" id="transcribe-btn" disabled>
                    📝 Transcribe Only
                </button>
                <button class="transcribe-btn diarize" id="transcribe-diarize-btn" disabled>
                    👥 Transcribe + Speakers
                </button>
            </div>
        `;

        this.cacheElements();
    }

    cacheElements() {
        this.elements = {
            transcribeBtn: this.container.querySelector('#transcribe-btn'),
            transcribeDiarizeBtn: this.container.querySelector('#transcribe-diarize-btn')
        };
    }

    setupEventListeners() {
        this.elements.transcribeBtn.addEventListener('click', () => {
            window.app.transcribe('transcribe');
        });

        this.elements.transcribeDiarizeBtn.addEventListener('click', () => {
            window.app.transcribe('diarize');
        });
    }

    enable() {
        this.elements.transcribeBtn.disabled = false;
        this.elements.transcribeDiarizeBtn.disabled = false;
        this.elements.transcribeBtn.textContent = '📝 Transcribe Only';
        this.elements.transcribeDiarizeBtn.textContent = '👥 Transcribe + Speakers';
    }

    disable() {
        this.elements.transcribeBtn.disabled = true;
        this.elements.transcribeDiarizeBtn.disabled = true;
        this.elements.transcribeBtn.textContent = 'Select audio file or record audio';
        this.elements.transcribeDiarizeBtn.textContent = 'Select audio file or record audio';
    }
}
