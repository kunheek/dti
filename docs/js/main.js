/**
 * Directional Textual Inversion - Project Page Scripts
 */

// Tab switching functionality
document.addEventListener('DOMContentLoaded', function () {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');

            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Subject tab switching
    const subjectButtons = document.querySelectorAll('.subject-btn');
    const subjectPanels = document.querySelectorAll('.subject-panel');

    subjectButtons.forEach(button => {
        button.addEventListener('click', () => {
            const subjectId = button.getAttribute('data-subject');

            // Remove active class from all buttons and panels
            subjectButtons.forEach(btn => btn.classList.remove('active'));
            subjectPanels.forEach(panel => panel.classList.remove('active'));

            // Add active class to clicked button and corresponding panel
            button.classList.add('active');
            document.getElementById(subjectId).classList.add('active');
        });
    });

    // Initialize horizontal gallery lightbox
    initHorizontalGalleryLightbox();
});

// Copy BibTeX to clipboard
function copyBibtex() {
    const bibtexCode = document.querySelector('.bibtex code');
    const text = bibtexCode.textContent;

    navigator.clipboard.writeText(text).then(() => {
        const copyBtn = document.querySelector('.copy-btn');
        const originalHTML = copyBtn.innerHTML;

        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        copyBtn.style.color = '#10b981';

        setTimeout(() => {
            copyBtn.innerHTML = originalHTML;
            copyBtn.style.color = '';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy: ', err);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
    });
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Horizontal Gallery Lightbox
function initHorizontalGalleryLightbox() {
    const galleries = document.querySelectorAll('.horizontal-gallery');

    galleries.forEach(gallery => {
        const items = gallery.querySelectorAll('.gallery-item');
        const images = Array.from(items).map(item => ({
            src: item.querySelector('img').src,
            caption: item.querySelector('figcaption')?.textContent || ''
        }));

        let currentIndex = 0;
        let lightbox = null;

        // Create lightbox element
        function createLightbox() {
            const overlay = document.createElement('div');
            overlay.className = 'lightbox-overlay';
            overlay.innerHTML = `
                <button class="lightbox-close" aria-label="Close">
                    <i class="fas fa-times"></i>
                </button>
                <button class="lightbox-nav prev" aria-label="Previous">
                    <i class="fas fa-chevron-left"></i>
                </button>
                <button class="lightbox-nav next" aria-label="Next">
                    <i class="fas fa-chevron-right"></i>
                </button>
                <div class="lightbox-content">
                    <img src="" alt="Gallery image">
                    <p class="lightbox-caption"></p>
                </div>
            `;
            document.body.appendChild(overlay);
            return overlay;
        }

        function openLightbox(index) {
            if (!lightbox) {
                lightbox = createLightbox();

                // Event listeners
                lightbox.querySelector('.lightbox-close').addEventListener('click', closeLightbox);
                lightbox.querySelector('.lightbox-nav.prev').addEventListener('click', showPrev);
                lightbox.querySelector('.lightbox-nav.next').addEventListener('click', showNext);
                lightbox.addEventListener('click', (e) => {
                    if (e.target === lightbox) closeLightbox();
                });
            }

            currentIndex = index;
            updateLightbox();
            lightbox.classList.add('active');
            document.body.style.overflow = 'hidden';

            // Keyboard navigation
            document.addEventListener('keydown', handleKeydown);
        }

        function closeLightbox() {
            if (lightbox) {
                lightbox.classList.remove('active');
                document.body.style.overflow = '';
                document.removeEventListener('keydown', handleKeydown);
            }
        }

        function updateLightbox() {
            const img = lightbox.querySelector('.lightbox-content img');
            const caption = lightbox.querySelector('.lightbox-caption');

            img.src = images[currentIndex].src;
            caption.textContent = images[currentIndex].caption;

            // Update navigation visibility
            const prevBtn = lightbox.querySelector('.lightbox-nav.prev');
            const nextBtn = lightbox.querySelector('.lightbox-nav.next');
            prevBtn.style.display = currentIndex === 0 ? 'none' : 'flex';
            nextBtn.style.display = currentIndex === images.length - 1 ? 'none' : 'flex';
        }

        function showPrev() {
            if (currentIndex > 0) {
                currentIndex--;
                updateLightbox();
            }
        }

        function showNext() {
            if (currentIndex < images.length - 1) {
                currentIndex++;
                updateLightbox();
            }
        }

        function handleKeydown(e) {
            if (e.key === 'Escape') closeLightbox();
            if (e.key === 'ArrowLeft') showPrev();
            if (e.key === 'ArrowRight') showNext();
        }

        // Attach click events to gallery items
        items.forEach((item, index) => {
            item.addEventListener('click', () => openLightbox(index));
        });
    });
}

// Lazy loading for images (optional enhancement)
document.addEventListener('DOMContentLoaded', function () {
    const images = document.querySelectorAll('img[data-src]');

    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                observer.unobserve(img);
            }
        });
    });

    images.forEach(img => imageObserver.observe(img));
});
