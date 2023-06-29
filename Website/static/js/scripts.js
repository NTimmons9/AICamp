window.addEventListener('DOMContentLoaded', event => {
    
    // Navbar shrink function
    var navbarShrink = function () {
        const navbarCollapsible = document.body.querySelector('#mainNav');
        if (!navbarCollapsible) {
            return;
        }
        if (window.scrollY === 0) {
            navbarCollapsible.classList.remove('navbar-shrink')
        } else {
            navbarCollapsible.classList.add('navbar-shrink')
        }
    };

    // Shrink the navbar 
    navbarShrink();

    // Shrink the navbar when page is scrolled
    document.addEventListener('scroll', navbarShrink);

    //  Activate Bootstrap scrollspy on the main nav element
    const mainNav = document.body.querySelector('#mainNav');
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#mainNav',
            rootMargin: '0px 0px -40%',
        });
    };

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

    // Handle file input change event
    const fileInput = document.getElementById('FileUploader');
    const fileLabel = document.querySelector('.btn.btn-primary.btn-x2.text-uppercase');
    
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            fileLabel.textContent = file.name;
        } else {
            fileLabel.textContent = 'Select image';
        }
    });

  const colorChanging = document.getElementById("colorChanging");

  var text = "{{dng}}";
  var colorClass = "";

  colorChanging.addEventListener('HTMLDetailsElement', () => {
    if (text === "No Threat Detected :)") {
      colorClass = "green";
    }
    else {
      colorClass = "red";
    }
  
    colorChanging.className = colorClass;
  });
});


