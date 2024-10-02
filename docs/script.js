window.addEventListener("DOMContentLoaded", function () {
    // getting all the sections and nav links
    const sections = document.querySelectorAll('section');
    const navItems = document.querySelectorAll('nav ul li');

    // Detecting the current section and adding active class to the nav link
    function activateSection() {
        let currentSection = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (pageYOffset >= sectionTop - 60) {
                currentSection = section.getAttribute('id');
            }
        });

        navItems.forEach(li => {
            li.classList.remove('active');
            let link = li.querySelector("a");
            if (link.getAttribute('href').includes(currentSection)) {
                li.classList.add('active');
            }
        });
    }

    // Event listener for scrolling
    window.addEventListener('scroll', activateSection);

    document.getElementById("hamburger").addEventListener("click", function () {
        let menu = document.querySelector("header");

        if (menu.classList.contains("open")) {
            menu.classList.remove("open");
        } else {
            menu.classList.add("open");
        }
    });

    document.getElementById("main-wrap").addEventListener("click", function () {
        document.querySelector("header").classList.remove("open");
    });
});
