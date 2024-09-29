// getting all the sections and nav links
const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('nav ul li a');

// Detecting the current section and adding active class to the nav link
function activateSection() {
    let currentSection = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (pageYOffset >= sectionTop - 60) {
            currentSection = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href').includes(currentSection)) {
            link.classList.add('active');
        }
    });
}

// Event listener for scrolling
window.addEventListener('scroll', activateSection);
