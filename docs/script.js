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

// this is to highlight all the best scores when the column or row is hovered over. Makes the data even more readable and easy to understand
// How it works (for my understanding):
// get all the th elements in the tr element with the class "score-heading" and log out their values (and index in the parent) when element is hovered over
// when that happens highlight the cells in the table of the same index as the th element hovered over with the class "best"
// when the mouse leaves the th element remove the highlight from the cells in the table

const scoreHeadings = document.querySelectorAll('tr.score-heading');
scoreHeadings.forEach(scoreHeading => {
    const thElements = scoreHeading.querySelectorAll('th');
    thElements.forEach(th => {
        th.addEventListener('mouseover', () => {
            console.log(th.textContent);
            const index = Array.from(th.parentElement.children).indexOf(th);
            // get the current table
            const table = th.closest('table');
            const tableRows = table.querySelectorAll('tr');
            tableRows.forEach(row => {
                const cells = row.querySelectorAll('td.best');
                cells.forEach((cell, i) => {
                    const cellIndex = Array.from(cell.parentElement.children).indexOf(cell);
                    if (cellIndex === index) {
                        cell.style.backgroundColor = '#FFF59D';
                        cell.style.boxShadow = '0 0 15px #FFF59D';
                    }
                });
            });
        });
    });
});

scoreHeadings.forEach(scoreHeading => {
    const thElements = scoreHeading.querySelectorAll('th');
    thElements.forEach(th => {
        th.addEventListener('mouseout', () => {
            console.log(th.textContent);
            const index = Array.from(th.parentElement.children).indexOf(th);
            const tableRows = document.querySelectorAll('table tr');
            tableRows.forEach(row => {
                const cells = row.querySelectorAll('td.best');
                cells.forEach((cell, i) => {
                    const cellIndex = Array.from(cell.parentElement.children).indexOf(cell);
                    if (cellIndex === index) {
                        cell.style.backgroundColor = '';
                        cell.style.boxShadow = '';
                    }
                });
            });
        });
    });
});




// Function to highlight table row cells with class "best"
function highlightBestCells(event) {
    const row = event.currentTarget;
    const bestCells = row.querySelectorAll('td.best');
    bestCells.forEach(cell => {
        cell.style.backgroundColor = '#FFF59D';
        cell.style.boxShadow = '0 0 15px #FFF59D';
    });
}

// Function to remove highlight from table row cells with class "best"
function removeHighlightBestCells(event) {
    const row = event.currentTarget;
    const bestCells = row.querySelectorAll('td.best');
    bestCells.forEach(cell => {
        cell.style.backgroundColor = '';
        cell.style.boxShadow = '';
    });
}

// Adding event listeners to each table row
const tableRows = document.querySelectorAll('table tr');
tableRows.forEach(row => {
    row.addEventListener('mouseover', highlightBestCells);
    row.addEventListener('mouseout', removeHighlightBestCells);
});