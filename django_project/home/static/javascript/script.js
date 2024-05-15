function openSidebar(){
    const sidebar = document.getElementById("mobile-navbar")
    sidebar.style.display = 'flex';
    setTimeout(() => {
        sidebar.style.width = '70%'; // Cambiamos el ancho después de un breve retraso para permitir que la pantalla se renderice primero
    }, 50);
}

function closeSidebar(){
    const sidebar = document.getElementById("mobile-navbar")
    sidebar.style.display = 'none';
    setTimeout(() => {
        sidebar.style.width = '0%'; // Cambiamos el ancho después de un breve retraso para permitir que la pantalla se renderice primero
    }, 50);
}