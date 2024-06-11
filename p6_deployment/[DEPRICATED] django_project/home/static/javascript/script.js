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


// ################################# PREDICCIONES.html ################################# 
// Calendario
document.addEventListener("DOMContentLoaded", function() {
    // Configurar el calendario
    flatpickr("#fechaSeleccionada", {
        dateFormat: "d-m-Y",
        minDate: "01-01-2024",
        maxDate: new Date().fp_incr(3),
        onChange: function(selectedDates, dateStr, instance) {
            console.log("Fecha seleccionada:", dateStr);
            // No es necesario realizar ningún cambio aquí, el valor se establecerá automáticamente
        }
    });

    // Obtener la fecha seleccionada del localStorage y enviarla al backend
    document.getElementById("fechaSeleccionada").addEventListener("change", function() {
        const fechaSelected = this.value;
        window.location.href = `/home/predicciones/?fechaSeleccionada=${fechaSelected}`;
    });
});