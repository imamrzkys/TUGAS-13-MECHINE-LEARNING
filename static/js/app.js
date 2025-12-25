document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            
            const spans = navToggle.querySelectorAll('span');
            if (navMenu.classList.contains('active')) {
                spans[0].style.transform = 'rotate(45deg) translateY(8px)';
                spans[1].style.opacity = '0';
                spans[2].style.transform = 'rotate(-45deg) translateY(-8px)';
            } else {
                spans[0].style.transform = 'none';
                spans[1].style.opacity = '1';
                spans[2].style.transform = 'none';
            }
        });
        
        document.addEventListener('click', function(e) {
            if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
                navMenu.classList.remove('active');
                const spans = navToggle.querySelectorAll('span');
                spans[0].style.transform = 'none';
                spans[1].style.opacity = '1';
                spans[2].style.transform = 'none';
            }
        });
    }
    
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            if (navMenu) {
                navMenu.classList.remove('active');
            }
        });
    });
    
    let lastScrollTop = 0;
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (scrollTop > lastScrollTop && scrollTop > 100) {
            navbar.style.transform = 'translateY(-100%)';
        } else {
            navbar.style.transform = 'translateY(0)';
        }
        
        if (scrollTop > 50) {
            navbar.style.background = 'rgba(10, 10, 15, 0.95)';
            navbar.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.3)';
        } else {
            navbar.style.background = 'rgba(10, 10, 15, 0.8)';
            navbar.style.boxShadow = 'none';
        }
        
        lastScrollTop = scrollTop;
    });
    
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(message => {
        setTimeout(() => {
            message.style.animation = 'slide-out-right 0.3s ease forwards';
            setTimeout(() => {
                message.remove();
            }, 300);
        }, 5000);
    });
    
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    const animatedElements = document.querySelectorAll('.feature-card, .stat-card, .about-card, .faq-card');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
    
    const smoothScrollLinks = document.querySelectorAll('a[href^="#"]');
    smoothScrollLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href !== '#' && href !== '') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
    
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(btn => {
        btn.addEventListener('mouseenter', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const ripple = document.createElement('span');
            ripple.style.position = 'absolute';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.style.width = '0';
            ripple.style.height = '0';
            ripple.style.borderRadius = '50%';
            ripple.style.background = 'rgba(255, 255, 255, 0.3)';
            ripple.style.transform = 'translate(-50%, -50%)';
            ripple.style.animation = 'ripple 0.6s ease-out';
            
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
    
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                width: 200px;
                height: 200px;
                opacity: 0;
            }
        }
        @keyframes slide-out-right {
            to {
                opacity: 0;
                transform: translateX(100%);
            }
        }
    `;
    document.head.appendChild(style);
    
    const cards = document.querySelectorAll('.glass-card, .feature-card, .stat-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;
            
            this.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-5px)`;
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0)';
        });
    });
    
    function createParticles() {
        const particleCount = 30;
        const container = document.querySelector('.aurora-background');
        
        if (!container) return;
        
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.style.position = 'absolute';
            particle.style.width = Math.random() * 3 + 1 + 'px';
            particle.style.height = particle.style.width;
            particle.style.background = `rgba(255, ${Math.random() * 100 + 107}, ${Math.random() * 100 + 157}, 0.3)`;
            particle.style.borderRadius = '50%';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animation = `float ${Math.random() * 10 + 10}s ease-in-out infinite`;
            particle.style.animationDelay = Math.random() * 5 + 's';
            particle.style.pointerEvents = 'none';
            
            container.appendChild(particle);
        }
        
        const floatStyle = document.createElement('style');
        floatStyle.textContent = `
            @keyframes float {
                0%, 100% {
                    transform: translate(0, 0);
                    opacity: 0;
                }
                10% {
                    opacity: 0.5;
                }
                50% {
                    transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px);
                    opacity: 1;
                }
                90% {
                    opacity: 0.5;
                }
            }
        `;
        document.head.appendChild(floatStyle);
    }
    
    createParticles();
    
    const counters = document.querySelectorAll('.stat-value');
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;
                const text = target.textContent;
                
                if (text.includes('%')) {
                    animateCounter(target, 0, 96, 2000, '%+');
                } else if (text.includes('K')) {
                    animateCounter(target, 0, 40, 2000, 'K+');
                } else if (text.includes('s')) {
                    target.textContent = '<1s';
                }
                
                counterObserver.unobserve(target);
            }
        });
    }, { threshold: 0.5 });
    
    counters.forEach(counter => counterObserver.observe(counter));
    
    function animateCounter(element, start, end, duration, suffix) {
        const startTime = performance.now();
        
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const current = Math.floor(start + (end - start) * easeOutQuart);
            
            element.textContent = current + suffix;
            
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        
        requestAnimationFrame(update);
    }
    
    const heroSection = document.querySelector('.hero-section');
    if (heroSection) {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            heroSection.style.transform = `translateY(${scrolled * 0.5}px)`;
            heroSection.style.opacity = 1 - scrolled / 700;
        });
    }
    
    document.querySelectorAll('img').forEach(img => {
        img.addEventListener('load', function() {
            this.style.animation = 'fade-in 0.5s ease';
        });
    });
    
    const ctaButtons = document.querySelectorAll('.btn-glow');
    ctaButtons.forEach(btn => {
        setInterval(() => {
            btn.style.animation = 'pulse-glow 2s ease-in-out';
            setTimeout(() => {
                btn.style.animation = '';
            }, 2000);
        }, 5000);
    });
    
    const pulseGlowStyle = document.createElement('style');
    pulseGlowStyle.textContent = `
        @keyframes pulse-glow {
            0%, 100% {
                box-shadow: 0 0 20px rgba(255, 107, 157, 0.3);
            }
            50% {
                box-shadow: 0 0 40px rgba(255, 107, 157, 0.6), 0 0 60px rgba(255, 165, 0, 0.4);
            }
        }
    `;
    document.head.appendChild(pulseGlowStyle);
    
    let mouseX = 0;
    let mouseY = 0;
    let cursorX = 0;
    let cursorY = 0;
    
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });
    
    function animateCursor() {
        const diffX = mouseX - cursorX;
        const diffY = mouseY - cursorY;
        
        cursorX += diffX * 0.1;
        cursorY += diffY * 0.1;
        
        const glowElements = document.querySelectorAll('.btn-glow:hover, .glass-card:hover');
        glowElements.forEach(el => {
            const rect = el.getBoundingClientRect();
            const x = cursorX - rect.left;
            const y = cursorY - rect.top;
            
            el.style.setProperty('--mouse-x', x + 'px');
            el.style.setProperty('--mouse-y', y + 'px');
        });
        
        requestAnimationFrame(animateCursor);
    }
    
    animateCursor();
    
    console.log('%c🌸 Aurora Fake News Detector 🌸', 'font-size: 20px; font-weight: bold; background: linear-gradient(135deg, #FF6B9D, #FFA500); -webkit-background-clip: text; color: transparent;');
    console.log('%cPowered by Machine Learning & Naive Bayes', 'font-size: 14px; color: #FF6B9D;');
    console.log('%cBuilt with Flask, Python, and ❤️', 'font-size: 12px; color: #FFA500;');
});
