(function () {
    "use strict";

    const locationButton = document.getElementById("refreshResponderLocation");
    const locationMessage = document.getElementById("responderLocationMessage");
    const locationFreshness = document.getElementById("responderLocationFreshness");

    function setMessage(element, message, isError) {
        if (!element) return;
        element.textContent = message;
        element.classList.toggle("is-error", Boolean(isError));
    }

    async function sendJson(url, method, payload) {
        const response = await fetch(url, {
            method,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.message || "The request could not be completed.");
        }
        return result;
    }

    if (locationButton) {
        locationButton.addEventListener("click", function () {
            if (!navigator.geolocation) {
                setMessage(locationMessage, "This browser does not support location sharing.", true);
                return;
            }

            locationButton.disabled = true;
            setMessage(locationMessage, "Acquiring your current location…", false);
            navigator.geolocation.getCurrentPosition(
                async function (position) {
                    const payload = {
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude,
                        accuracy: position.coords.accuracy
                    };
                    try {
                        const result = await sendJson("/api/responder/location", "POST", payload);
                        if (locationFreshness) {
                            locationFreshness.textContent = "Updated just now";
                        }
                        setMessage(locationMessage, result.message, false);
                    } catch (error) {
                        setMessage(locationMessage, error.message, true);
                    } finally {
                        locationButton.disabled = false;
                    }
                },
                function (error) {
                    const messages = {
                        1: "Location permission denied. Allow location access and try again.",
                        2: "Your current location is unavailable.",
                        3: "Location request timed out. Try again."
                    };
                    setMessage(
                        locationMessage,
                        messages[error.code] || "Unable to determine your current location.",
                        true
                    );
                    locationButton.disabled = false;
                },
                {
                    enableHighAccuracy: true,
                    maximumAge: 0,
                    timeout: 20000
                }
            );
        });
    }

    document.querySelectorAll(".responder-assignment-action").forEach(function (button) {
        button.addEventListener("click", async function () {
            const assignmentId = button.dataset.assignmentId;
            const nextStatus = button.dataset.nextStatus;
            button.disabled = true;
            try {
                await sendJson(
                    "/api/responder/assignments/" + encodeURIComponent(assignmentId) + "/status",
                    "PUT",
                    { status: nextStatus }
                );
                window.location.reload();
            } catch (error) {
                setMessage(locationMessage, error.message, true);
                button.disabled = false;
            }
        });
    });
}());
