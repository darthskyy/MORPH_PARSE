window.addEventListener("DOMContentLoaded", function () {
    // getting all the sections and nav links
    const sections = document.querySelectorAll('section');
    const navItems = document.querySelectorAll('nav ul li');
    const sectionIds = Array.from(sections).map(section => section.getAttribute('id'));
    // six colors for the to cycle through for the background color of the sections
    const scrollColors = true;
    const colors = ['#EEF2FB', '#ADC0EB', '#6B8DDB', '#5B80D7', '#3A67CF', '#244594'];
    

    // Detecting the current section and adding active class to the nav link
    function activateSection() {
        let currentSection = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (pageYOffset >= sectionTop - 300) {
                console.log(sectionTop);
                console.log(pageYOffset);
                console.log(section.getAttribute('id'));
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
        
        if (!scrollColors) return;
        // Changing the background color of the section
        const sectionIndex = sectionIds.indexOf(currentSection);
        document.body.style.backgroundColor = colors[sectionIndex];
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
