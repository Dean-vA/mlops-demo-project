// Audio Notification Component (js/components/audio-notification.js)
class AudioNotificationComponent {
    constructor() {
        this.container = document.getElementById('audio-notification-component');
        this.hideTimer = null;
    }

    async init() {
        this.render();
        this.setupEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="audio-notification" id="audio-notification">🔊 Playing segment...</div>
        `;

        this.cacheElements();
    }

    cacheElements() {
        this.elements = {
            audioNotification: this.container.querySelector('#audio-notification')
        };
    }

    setupEventListeners() {
        document.addEventListener('segmentPlaybackStarted', (event) => {
            const { startTime, endTime } = event.detail;
            this.show(`🔊 Playing segment (${startTime.toFixed(1)}s - ${endTime.toFixed(1)}s)`);
        });

        document.addEventListener('segmentPlaybackStopped', () => {
            this.hide();
        });
    }

    show(message) {
        this.elements.audioNotification.textContent = message;
        this.elements.audioNotification.classList.add('show');

        // Auto-hide after 3 seconds
        if (this.hideTimer) clearTimeout(this.hideTimer);
        this.hideTimer = setTimeout(() => this.hide(), 3000);
    }

    hide() {
        this.elements.audioNotification.classList.remove('show');
        if (this.hideTimer) {
            clearTimeout(this.hideTimer);
            this.hideTimer = null;
        }
    }
}
