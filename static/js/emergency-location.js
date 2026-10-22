(function () {
    "use strict";

    const api = {
        start: "/api/emergency/start",
        update: "/api/emergency/location",
        stop: "/api/emergency/stop",
        active: "/api/emergency/active"
    };

    async function postJson(url, payload) {
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.message || "The request could not be completed.");
        }
        return result;
    }

    function showNotice(message, isError) {
        const notice = document.getElementById("emergencyNotice");
        if (!notice) return;
        notice.textContent = message;
        notice.classList.toggle("is-error", Boolean(isError));
        notice.classList.toggle("is-visible", Boolean(message));
    }

    function normalizePosition(position) {
        const latitude = Number(position.coords.latitude);
        const longitude = Number(position.coords.longitude);
        const accuracy = Number(position.coords.accuracy);

        console.log("GPS timestamp:", position.timestamp);
        console.log("GPS latitude:", latitude);
        console.log("GPS longitude:", longitude);
        console.log("GPS accuracy:", accuracy);

        if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90) {
            showNotice("Unable to determine your current location.", true);
            return null;
        }

        if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
            showNotice("Unable to determine your current location.", true);
            return null;
        }

        if (!Number.isFinite(accuracy) || accuracy < 0) {
            showNotice("GPS location is currently unavailable.", true);
            return null;
        }

        return {
            latitude,
            longitude,
            accuracy
        };
    }

    function geoErrorMessage(error) {
        if (error && error.code === 1) {
            return "Location permission denied. Please allow location access.";
        }
        if (error && error.code === 2) {
            return "GPS location is currently unavailable.";
        }
        if (error && error.code === 3) {
            return "GPS acquisition timed out. Retrying...";
        }
        return "Unable to determine your current location.";
    }

    function createMap(element, latitude, longitude, zoom) {
        if (!element || !window.L) return null;
        const map = L.map(element).setView([latitude, longitude], zoom);
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "&copy; OpenStreetMap contributors",
            maxZoom: 19
        }).addTo(map);
        return map;
    }

    function initializeNearestResponders() {
        const list = document.getElementById("nearestRespondersList");
        const emptyState = document.getElementById("nearestRespondersEmpty");
        const config = document.getElementById("emergencyPageConfig");
        if (!list || !emptyState || !config) return null;

        function setEmptyMessage(message) {
            list.replaceChildren();
            emptyState.textContent = message;
            emptyState.hidden = false;
        }

        function readCoords() {
            const latitudeValue = config.dataset.activeLatitude;
            const longitudeValue = config.dataset.activeLongitude;
            if (latitudeValue === "" || longitudeValue === "") {
                return null;
            }
            const latitude = Number(latitudeValue);
            const longitude = Number(longitudeValue);
            if (
                !Number.isFinite(latitude) ||
                latitude < -90 ||
                latitude > 90 ||
                !Number.isFinite(longitude) ||
                longitude < -180 ||
                longitude > 180
            ) return null;
            return { latitude, longitude };
        }

        let lastRefreshAt = 0;
        async function refresh() {
            const coords = readCoords();
            if (!coords) {
                setEmptyMessage("Start or resume emergency tracking to view nearby responders.");
                return;
            }

            if (Date.now() - lastRefreshAt < 5000) return;
            lastRefreshAt = Date.now();

            try {
                const response = await fetch(
                    "/api/responders/nearby?latitude=" + coords.latitude + "&longitude=" + coords.longitude + "&limit=3",
                    { headers: { "Accept": "application/json" } }
                );
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.message || "Unable to load nearby responders.");
                }

                const responders = result.responders || [];
                list.replaceChildren();
                if (!responders.length) {
                    emptyState.textContent = "No available responders are currently nearby.";
                    emptyState.hidden = false;
                    return;
                }

                emptyState.hidden = true;
                responders.forEach(function (responder) {
                    const item = document.createElement("article");
                    item.className = "nearest-responder-item";

                    const icon = document.createElement("div");
                    icon.className = "nearest-responder-badge";
                    icon.textContent = (responder.responder_type || "R").charAt(0).toUpperCase();

                    const metadata = document.createElement("div");
                    metadata.className = "nearest-responder-meta";

                    const name = document.createElement("strong");
                    name.textContent = responder.name || "Responder";
                    const type = document.createElement("span");
                    type.textContent = responder.responder_type || "Responder";
                    const distance = document.createElement("small");
                    const distanceMeters = Number(responder.distance_meters);
                    distance.textContent = Number.isFinite(distanceMeters)
                        ? Math.round(distanceMeters) + " m away"
                        : "Distance unavailable";

                    metadata.append(name, type, distance);

                    const status = document.createElement("span");
                    status.className = "nearby-status";
                    status.textContent = responder.availability_status || "Available";

                    item.append(icon, metadata, status);
                    list.appendChild(item);
                });
            } catch (error) {
                console.error("Unable to load responders:", error);
                setEmptyMessage("Unable to load nearby responders right now.");
            }
        }

        refresh();
        window.setInterval(refresh, 15000);
        return { refresh };
    }

    function popupContent(incident) {
        const content = document.createElement("div");
        const title = document.createElement("strong");
        title.textContent = incident.emergency_type || "Emergency";
        const details = document.createElement("div");
        details.textContent = incident.person_name || "Person not specified";
        const coordinates = document.createElement("small");
        coordinates.textContent =
            Number(incident.latitude).toFixed(5) + ", " +
            Number(incident.longitude).toFixed(5);
        content.append(title, details, coordinates);
        return content;
    }

    function initializeResponderDirectory() {
        const list = document.getElementById("responderDirectoryList");
        const message = document.getElementById("responderDirectoryMessage");
        const count = document.getElementById("responderDirectoryCount");
        const context = document.getElementById("responderDistanceContext");
        const nearbyButton = document.getElementById("findNearbyRespondersButton");
        const nearbyLocation = document.getElementById("nearbyResponderLocation");
        const nearbyList = document.getElementById("nearbyResponderList");
        const nearbyMessage = document.getElementById("nearbyResponderMessage");
        const filterButtons = Array.from(
            document.querySelectorAll("[data-responder-filter]")
        );
        if (!list || !message || !count || !filterButtons.length) return;

        let responders = [];
        let nearbyResponders = [];
        let nearbyOrigin = null;
        let selectedType = "all";

        function activeEmergencyLocation() {
            const row = document.querySelector(
                "#responderIncidentList .active-incident[data-latitude][data-longitude]"
            );
            if (!row) return null;
            const latitude = Number(row.dataset.latitude);
            const longitude = Number(row.dataset.longitude);
            if (
                !Number.isFinite(latitude) ||
                latitude < -90 ||
                latitude > 90 ||
                !Number.isFinite(longitude) ||
                longitude < -180 ||
                longitude > 180
            ) return null;
            return { latitude, longitude };
        }

        function matchesSelectedType(responder) {
            if (selectedType === "all") return true;
            const type = String(responder.responder_type || "").toLowerCase();
            if (selectedType === "medical") {
                return type.includes("doctor") || type.includes("medical");
            }
            if (selectedType === "fire") {
                return type.includes("fire") || type.includes("rescue");
            }
            return type.includes(selectedType);
        }

        function renderNearbyResponders() {
            if (!nearbyList || !nearbyMessage) return;
            nearbyList.replaceChildren();

            if (!nearbyOrigin) {
                nearbyMessage.hidden = false;
                return;
            }

            const visibleResponders = nearbyResponders.filter(matchesSelectedType);
            if (!visibleResponders.length) {
                nearbyMessage.textContent = nearbyResponders.length
                    ? "No nearby responders match the selected service type."
                    : "No available responders with valid locations were found near this location.";
                nearbyMessage.hidden = false;
                return;
            }

            nearbyMessage.hidden = true;
            visibleResponders.forEach(function (responder) {
                const item = document.createElement("article");
                item.className = "responder-directory-item";

                const details = document.createElement("div");
                details.className = "responder-directory-details";
                const name = document.createElement("strong");
                name.textContent = responder.name || "Unnamed responder";
                const type = document.createElement("span");
                type.textContent = responder.responder_type || "Responder";
                const phone = document.createElement("span");
                phone.textContent = responder.phone || "No phone number";
                const meters = Number(responder.distance_meters);
                const distance = document.createElement("small");
                distance.textContent = Number.isFinite(meters)
                    ? meters < 1000
                        ? Math.round(meters) + " m away"
                        : (meters / 1000).toFixed(1) + " km away"
                    : "Distance unavailable";
                details.append(name, type, phone, distance);

                const status = document.createElement("span");
                status.className = "responder-status status-available";
                status.textContent = responder.availability_status || "Available";

                const actions = document.createElement("div");
                actions.className = "responder-directory-actions";
                const phoneNumber = String(responder.phone || "").trim();
                if (phoneNumber) {
                    const callLink = document.createElement("a");
                    callLink.className = "secondary-btn responder-action";
                    callLink.textContent = "Call";
                    callLink.href = "tel:" + phoneNumber.replace(/[^\d+*#,;]/g, "");
                    actions.appendChild(callLink);
                }

                const latitude = Number(responder.latitude);
                const longitude = Number(responder.longitude);
                if (
                    Number.isFinite(latitude) &&
                    latitude >= -90 && latitude <= 90 &&
                    Number.isFinite(longitude) &&
                    longitude >= -180 && longitude <= 180
                ) {
                    const directionsUrl =
                        "https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route=" +
                        nearbyOrigin.latitude + "%2C" + nearbyOrigin.longitude + "%3B" +
                        latitude + "%2C" + longitude;
                    actions.appendChild(buildAction(
                        "Directions",
                        directionsUrl,
                        "secondary-btn responder-action"
                    ));
                }

                item.append(details, status, actions);
                nearbyList.appendChild(item);
            });
        }

        function getDeviceLocation() {
            return new Promise(function (resolve, reject) {
                if (!navigator.geolocation) {
                    reject(new Error("This browser does not support device location."));
                    return;
                }
                if (!window.isSecureContext) {
                    reject(new Error("Chrome requires HTTPS for device location on phones. Use localhost for local testing."));
                    return;
                }

                navigator.geolocation.getCurrentPosition(
                    function (position) {
                        const latitude = Number(position.coords.latitude);
                        const longitude = Number(position.coords.longitude);
                        const accuracy = Number(position.coords.accuracy);
                        if (
                            !Number.isFinite(latitude) || latitude < -90 || latitude > 90 ||
                            !Number.isFinite(longitude) || longitude < -180 || longitude > 180 ||
                            !Number.isFinite(accuracy) || accuracy < 0
                        ) {
                            reject(new Error("The device returned invalid location coordinates."));
                            return;
                        }
                        resolve({ latitude, longitude, accuracy, source: "device" });
                    },
                    function (error) {
                        if (error.code === 1) {
                            reject(new Error("Location permission denied. Allow Chrome location access and retry."));
                        } else if (error.code === 2) {
                            reject(new Error("Device location is currently unavailable."));
                        } else if (error.code === 3) {
                            reject(new Error("Location acquisition timed out. Please retry."));
                        } else {
                            reject(new Error("Unable to determine device location."));
                        }
                    },
                    {
                        enableHighAccuracy: true,
                        maximumAge: 0,
                        timeout: 20000
                    }
                );
            });
        }

        async function findNearbyResponders() {
            if (!nearbyButton || !nearbyMessage || !nearbyLocation) return;

            nearbyButton.disabled = true;
            nearbyMessage.textContent = "Getting location and searching nearby available responders…";
            nearbyMessage.hidden = false;
            nearbyList.replaceChildren();

            try {
                const incidentOrigin = activeEmergencyLocation();
                let origin;
                if (incidentOrigin) {
                    origin = Object.assign({}, incidentOrigin, { source: "emergency" });
                    nearbyLocation.textContent =
                        "Searching from active emergency: " +
                        origin.latitude.toFixed(5) + ", " + origin.longitude.toFixed(5) + ".";
                } else {
                    nearbyLocation.textContent = "Requesting this device's current location…";
                    origin = await getDeviceLocation();
                    nearbyLocation.textContent =
                        "Searching from device location: " +
                        origin.latitude.toFixed(5) + ", " + origin.longitude.toFixed(5) +
                        " (accuracy ±" + Math.round(origin.accuracy) + " m).";
                }

                const query = new URLSearchParams({
                    latitude: String(origin.latitude),
                    longitude: String(origin.longitude),
                    limit: "5"
                });
                const response = await fetch("/api/responders/nearby?" + query.toString(), {
                    headers: { "Accept": "application/json" }
                });
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.message || "Unable to search nearby responders.");
                }

                nearbyOrigin = origin;
                nearbyResponders = result.responders || [];
                renderNearbyResponders();
            } catch (error) {
                nearbyOrigin = null;
                nearbyResponders = [];
                nearbyLocation.textContent = error.message;
                nearbyMessage.textContent = "Nearby responders could not be loaded.";
                nearbyMessage.hidden = false;
                console.error("Unable to find nearby responders:", error);
            } finally {
                nearbyButton.disabled = false;
            }
        }

        function buildAction(label, href, className) {
            const link = document.createElement("a");
            link.className = className;
            link.textContent = label;
            link.href = href;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            return link;
        }

        function render() {
            const origin = activeEmergencyLocation();
            if (context) {
                context.textContent = origin
                    ? "Distances are measured from the nearest active emergency location."
                    : "No active emergency location is available; distances cannot be calculated yet.";
            }

            const visibleResponders = responders
                .filter(matchesSelectedType)
                .map(function (responder) {
                    const latitude = Number(responder.latitude);
                    const longitude = Number(responder.longitude);
                    const hasLocation = Number.isFinite(latitude) &&
                        latitude >= -90 && latitude <= 90 &&
                        Number.isFinite(longitude) &&
                        longitude >= -180 && longitude <= 180;
                    const distance = origin && hasLocation
                        ? distanceMeters(origin.latitude, origin.longitude, latitude, longitude)
                        : null;
                    return { responder, distance };
                })
                .sort(function (left, right) {
                    const leftAvailable = String(left.responder.availability_status).toLowerCase() === "available";
                    const rightAvailable = String(right.responder.availability_status).toLowerCase() === "available";
                    if (leftAvailable !== rightAvailable) return leftAvailable ? -1 : 1;
                    if (left.distance !== null && right.distance !== null && left.distance !== right.distance) {
                        return left.distance - right.distance;
                    }
                    return String(left.responder.name || "").localeCompare(String(right.responder.name || ""));
                });

            list.replaceChildren();
            count.textContent = visibleResponders.length + " responder" +
                (visibleResponders.length === 1 ? "" : "s");
            if (!visibleResponders.length) {
                message.textContent = responders.length
                    ? "No registered responders match this type."
                    : "No responders have been registered yet.";
                message.hidden = false;
                return;
            }

            message.hidden = true;
            visibleResponders.forEach(function (entry) {
                const responder = entry.responder;
                const item = document.createElement("article");
                item.className = "responder-directory-item";

                const details = document.createElement("div");
                details.className = "responder-directory-details";
                const name = document.createElement("strong");
                name.textContent = responder.name || "Unnamed responder";
                const type = document.createElement("span");
                type.textContent = responder.responder_type || "Responder";
                const phone = document.createElement("span");
                phone.textContent = responder.phone || "No phone number";
                const distance = document.createElement("small");
                distance.textContent = entry.distance === null
                    ? "Distance unavailable"
                    : entry.distance < 1000
                        ? Math.round(entry.distance) + " m from incident"
                        : (entry.distance / 1000).toFixed(1) + " km from incident";
                details.append(name, type, phone, distance);

                const statusText = String(responder.availability_status || "Offline");
                const status = document.createElement("span");
                status.className = "responder-status status-" + statusText.toLowerCase();
                status.textContent = statusText;

                const actions = document.createElement("div");
                actions.className = "responder-directory-actions";
                const phoneNumber = String(responder.phone || "").trim();
                if (phoneNumber) {
                    const callLink = document.createElement("a");
                    callLink.className = "secondary-btn responder-action";
                    callLink.textContent = "Call";
                    callLink.href = "tel:" + phoneNumber.replace(/[^\d+*#,;]/g, "");
                    actions.appendChild(callLink);
                }

                const latitude = Number(responder.latitude);
                const longitude = Number(responder.longitude);
                if (
                    Number.isFinite(latitude) &&
                    latitude >= -90 && latitude <= 90 &&
                    Number.isFinite(longitude) &&
                    longitude >= -180 && longitude <= 180
                ) {
                    let mapUrl = "https://www.openstreetmap.org/?mlat=" + latitude +
                        "&mlon=" + longitude + "#map=16/" + latitude + "/" + longitude;
                    let mapLabel = "View location";
                    const origin = activeEmergencyLocation();
                    if (origin) {
                        mapUrl = "https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route=" +
                            origin.latitude + "%2C" + origin.longitude + "%3B" +
                            latitude + "%2C" + longitude;
                        mapLabel = "Directions";
                    }
                    actions.appendChild(buildAction(mapLabel, mapUrl, "secondary-btn responder-action"));
                }

                item.append(details, status, actions);
                list.appendChild(item);
            });
        }

        async function loadResponders() {
            try {
                const response = await fetch("/api/responders", {
                    headers: { "Accept": "application/json" }
                });
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.message || "Unable to load responders.");
                }
                responders = result.responders || [];
                render();
            } catch (error) {
                console.error("Unable to load responder directory:", error);
                count.textContent = "Unavailable";
                list.replaceChildren();
                message.textContent = "Unable to load registered responders. Please refresh the page and try again.";
                message.hidden = false;
            }
        }

        filterButtons.forEach(function (button) {
            button.addEventListener("click", function () {
                selectedType = String(button.dataset.responderFilter || "all").toLowerCase();
                filterButtons.forEach(function (filterButton) {
                    const selected = filterButton === button;
                    filterButton.classList.toggle("is-selected", selected);
                    filterButton.setAttribute("aria-pressed", String(selected));
                });
                render();
                renderNearbyResponders();
            });
        });

        if (nearbyButton) {
            nearbyButton.addEventListener("click", findNearbyResponders);
        }

        loadResponders();
        window.setInterval(function () {
            loadResponders();
        }, 15000);
        return { render };
    }

    function distanceMeters(latitudeA, longitudeA, latitudeB, longitudeB) {
        const earthRadius = 6371000;
        const latitudeDelta = (latitudeB - latitudeA) * Math.PI / 180;
        const longitudeDelta = (longitudeB - longitudeA) * Math.PI / 180;
        const haversine = Math.sin(latitudeDelta / 2) ** 2 +
            Math.cos(latitudeA * Math.PI / 180) *
            Math.cos(latitudeB * Math.PI / 180) *
            Math.sin(longitudeDelta / 2) ** 2;
        return earthRadius * 2 * Math.atan2(
            Math.sqrt(haversine),
            Math.sqrt(1 - haversine)
        );
    }

    function initializeOwnerTracking() {
        const config = document.getElementById("emergencyPageConfig");
        const form = document.getElementById("emergencyForm");
        if (!config || (!form && !document.getElementById("activeEmergencyPanel"))) return;

        const formPanel = document.getElementById("emergencyFormPanel");
        const activePanel = document.getElementById("activeEmergencyPanel");
        const startButton = document.getElementById("startTrackingButton");
        const checkLocationButton = document.getElementById("checkLocationButton");
        const currentLocationStatus = document.getElementById("currentLocationStatus");
        const resumeButton = document.getElementById("resumeTrackingButton");
        const stopButton = document.getElementById("stopTrackingButton");
        const mapElement = document.getElementById("emergencyMap");
        const coordinateText = document.getElementById("trackingCoordinates");
        const accuracyText = document.getElementById("trackingAccuracy");
        const nearestRespondersWidget = initializeNearestResponders();
        let incidentId = Number(config.dataset.activeId) || null;
        let watchId = null;
        let incidentMap = null;
        let locationMarker = null;
        let creatingIncident = false;
        let bestLocation = null;

        if (checkLocationButton) {
            checkLocationButton.addEventListener("click", function () {
                if (!navigator.geolocation) {
                    const message = "This browser does not support location sharing.";
                    if (currentLocationStatus) currentLocationStatus.textContent = message;
                    showNotice(message, true);
                    return;
                }

                checkLocationButton.disabled = true;
                if (currentLocationStatus) {
                    currentLocationStatus.textContent = "Acquiring precise location...";
                }

                navigator.geolocation.getCurrentPosition(
                    function (position) {
                        checkLocationButton.disabled = false;
                        const location = normalizePosition(position);
                        if (!location) {
                            if (currentLocationStatus) {
                                currentLocationStatus.textContent = "Unable to determine your current location.";
                            }
                            return;
                        }

                        config.dataset.activeLatitude = String(location.latitude);
                        config.dataset.activeLongitude = String(location.longitude);
                        if (currentLocationStatus) {
                            currentLocationStatus.textContent =
                                "Device location: " + location.latitude.toFixed(5) + ", " +
                                location.longitude.toFixed(5) + " (accuracy ±" +
                                Math.round(location.accuracy) + " m)";
                        }
                        if (nearestRespondersWidget) {
                            nearestRespondersWidget.refresh();
                        }
                        showNotice("Current device location acquired.", false);
                    },
                    function (error) {
                        checkLocationButton.disabled = false;
                        const message = geoErrorMessage(error);
                        if (currentLocationStatus) currentLocationStatus.textContent = message;
                        showNotice(message, true);
                    },
                    {
                        enableHighAccuracy: true,
                        maximumAge: 0,
                        timeout: 20000
                    }
                );
            });
        }

        function getAccuracyLabel(accuracy) {
            if (!Number.isFinite(accuracy)) return "GPS quality unknown";
            if (accuracy <= 20) return "Excellent GPS";
            if (accuracy <= 50) return "Good GPS";
            if (accuracy <= 100) return "Moderate GPS";
            return "Low quality GPS";
        }

        function setIncidentMap(latitude, longitude) {
            if (!incidentMap) {
                incidentMap = createMap(mapElement, latitude, longitude, 16);
            }
            if (!incidentMap) return;
            if (!locationMarker) {
                locationMarker = L.circleMarker([latitude, longitude], {
                    radius: 9,
                    color: "#ffffff",
                    weight: 3,
                    fillColor: "#d94a4a",
                    fillOpacity: 1
                }).addTo(incidentMap);
            } else {
                locationMarker.setLatLng([latitude, longitude]);
            }
            incidentMap.setView([latitude, longitude], 16);
        }

        function receivePosition(position, shouldCreateIncident, incidentDetails) {
            const location = normalizePosition(position);
            if (!location) return;

            if (bestLocation && location.accuracy > bestLocation.accuracy && bestLocation.accuracy <= 30 && location.accuracy > 80) {
                console.log("Ignoring lower-quality GPS update:", location.accuracy, "m vs previous", bestLocation.accuracy, "m");
                return;
            }

            bestLocation = {
                latitude: location.latitude,
                longitude: location.longitude,
                accuracy: location.accuracy,
                timestamp: position.timestamp
            };

            const updateDisplay = function () {
                coordinateText.textContent =
                    location.latitude.toFixed(5) + ", " +
                    location.longitude.toFixed(5);
                if (accuracyText) {
                    accuracyText.textContent = "±" + Math.round(location.accuracy) + " m (" + getAccuracyLabel(location.accuracy) + ")";
                }
                setIncidentMap(location.latitude, location.longitude);
                if (coordinateText) {
                    coordinateText.title = getAccuracyLabel(location.accuracy) + " • ±" + Math.round(location.accuracy) + " m";
                }

                config.dataset.activeLatitude = String(location.latitude);
                config.dataset.activeLongitude = String(location.longitude);
                if (nearestRespondersWidget) {
                    nearestRespondersWidget.refresh();
                }
            };

            if (shouldCreateIncident) {
                postJson(api.start, Object.assign({}, incidentDetails, location))
                    .then(function (result) {
                        creatingIncident = false;
                        incidentId = result.incident_id;
                        config.dataset.activeId = String(incidentId);
                        formPanel.hidden = true;
                        activePanel.hidden = false;
                        document.getElementById("trackingPerson").textContent = incidentDetails.person_name;
                        document.getElementById("trackingTitle").textContent = incidentDetails.emergency_type;
                        updateDisplay();
                        const assignment = result.assignment || {};
                        if (assignment.status === "assigned") {
                            showNotice("Emergency started. An assignment is recorded for a responder. No phone notification was sent.", false);
                        } else if (assignment.status === "no_eligible_responder") {
                            showNotice("Emergency started and location sharing is active, but no suitable responder with a recent location is available. You can retry assignment.", false);
                        } else {
                            showNotice("Emergency started. Your live location is being shared.", false);
                        }
                        const dispatchStatus = document.getElementById("assignmentDispatchStatus");
                        if (dispatchStatus) {
                            dispatchStatus.textContent = assignment.status === "assigned"
                                ? "Assignment recorded. The responder can view it after signing in; no phone notification was sent."
                                : assignment.status === "no_eligible_responder"
                                    ? "No eligible responder was available at dispatch time."
                                    : "";
                        }
                        const retryDispatch = document.getElementById("retryEmergencyDispatch");
                        if (retryDispatch) retryDispatch.hidden = false;
                        startWatching(false, null);
                    })
                    .catch(function (error) {
                        creatingIncident = false;
                        showNotice(error.message, true);
                        if (startButton) startButton.disabled = false;
                    });
                return;
            }

            if (!incidentId) {
                showNotice("No active incident is available for location sharing.", true);
                return;
            }
            postJson(api.update, Object.assign({ incident_id: incidentId }, location))
                .then(updateDisplay)
                .catch(function (error) {
                    showNotice(error.message, true);
                });
        }

        function startWatching(shouldCreateIncident, incidentDetails) {
            if (!navigator.geolocation) {
                showNotice("This browser does not support location sharing.", true);
                if (startButton) startButton.disabled = false;
                if (resumeButton) resumeButton.disabled = false;
                return;
            }
            bestLocation = null;
            if (watchId !== null) {
                navigator.geolocation.clearWatch(watchId);
            }
            watchId = navigator.geolocation.watchPosition(
                function (position) {
                    if (shouldCreateIncident) {
                        if (creatingIncident) return;
                        creatingIncident = true;
                        shouldCreateIncident = false;
                        receivePosition(position, true, incidentDetails);
                        return;
                    }
                    receivePosition(position, false, incidentDetails);
                },
                function (error) {
                    showNotice(geoErrorMessage(error), true);
                    if (watchId !== null) {
                        navigator.geolocation.clearWatch(watchId);
                        watchId = null;
                    }
                    if (startButton) startButton.disabled = false;
                    if (resumeButton) resumeButton.disabled = false;
                },
                {
                    enableHighAccuracy: true,
                    maximumAge: 0,
                    timeout: 20000
                }
            );
            showNotice("Acquiring precise location...", false);
        }

        if (config.dataset.activeId) {
            const latitude = Number(config.dataset.activeLatitude);
            const longitude = Number(config.dataset.activeLongitude);
            if (Number.isFinite(latitude) && Number.isFinite(longitude)) {
                setIncidentMap(latitude, longitude);
            }
        }

        if (form) {
            form.addEventListener("submit", function (event) {
                event.preventDefault();
                if (!form.reportValidity()) return;
                if (startButton) startButton.disabled = true;
                const incidentDetails = {
                    person_name: document.getElementById("personName").value.trim(),
                    contact_number: document.getElementById("phone").value.trim(),
                    emergency_type: document.getElementById("emergencyType").value,
                    location: document.getElementById("location").value.trim(),
                    description: document.getElementById("description").value.trim()
                };
                startWatching(true, incidentDetails);
            });
        }

        if (resumeButton) {
            resumeButton.addEventListener("click", function () {
                startWatching(false, null);
                resumeButton.disabled = true;
            });
        }

        if (stopButton) {
            stopButton.addEventListener("click", async function () {
                if (!incidentId) return;
                stopButton.disabled = true;
                try {
                    await postJson(api.stop, { incident_id: incidentId });
                    if (watchId !== null) {
                        navigator.geolocation.clearWatch(watchId);
                        watchId = null;
                    }
                    window.location.reload();
                } catch (error) {
                    showNotice(error.message, true);
                    stopButton.disabled = false;
                }
            });
        }

        const retryDispatchButton = document.getElementById("retryEmergencyDispatch");
        if (retryDispatchButton) {
            retryDispatchButton.addEventListener("click", async function () {
                if (!incidentId) return;
                retryDispatchButton.disabled = true;
                try {
                    const result = await postJson(
                        "/api/emergency/" + encodeURIComponent(incidentId) + "/dispatch",
                        {}
                    );
                    const dispatchStatus = document.getElementById("assignmentDispatchStatus");
                    if (dispatchStatus) {
                        dispatchStatus.textContent = result.assignment.status === "assigned"
                            ? "Assignment recorded. The responder can view it after signing in; no phone notification was sent."
                            : result.assignment.status === "already_assigned"
                                ? "This emergency already has an active assignment."
                                : "No eligible responder with a recent location is available right now.";
                    }
                } catch (error) {
                    showNotice(error.message, true);
                } finally {
                    retryDispatchButton.disabled = false;
                }
            });
        }
    }

    function initializeIncidentFeed(mapId, listId, countId) {
        const mapElement = document.getElementById(mapId);
        const list = document.getElementById(listId);
        const count = document.getElementById(countId);
        if (!mapElement || !list || !count) return;

        mapElement.hidden = !list.querySelector(".active-incident");
        let incidentMap = mapElement.hidden
            ? null
            : createMap(mapElement, 20, 0, 2);
        const markers = new Map();

        function createIncidentRow(incident) {
            const article = document.createElement("article");
            article.className = "active-incident";
            article.dataset.incidentId = String(incident.id);
            article.dataset.latitude = String(incident.latitude);
            article.dataset.longitude = String(incident.longitude);

            const badge = document.createElement("div");
            badge.className = "incident-type-icon";
            badge.textContent = "!";

            const info = document.createElement("div");
            info.className = "active-incident-copy";
            const title = document.createElement("strong");
            title.textContent = incident.emergency_type || "Emergency";
            const details = document.createElement("span");
            details.textContent = (incident.person_name || "Person not specified") +
                " · " + (incident.contact_number || "No contact provided");
            info.append(title, details);

            const location = document.createElement("div");
            location.className = "active-incident-location";
            const coordinates = document.createElement("span");
            coordinates.textContent = Number(incident.latitude).toFixed(5) +
                ", " + Number(incident.longitude).toFixed(5);
            location.appendChild(coordinates);
            if (incident.accuracy !== null && incident.accuracy !== undefined) {
                const accuracy = document.createElement("small");
                accuracy.textContent = "GPS accuracy ±" + Math.round(incident.accuracy) + " m";
                location.appendChild(accuracy);
            }

            article.append(badge, info, location);
            if (mapId === "responderMap") {
                const link = document.createElement("a");
                const lat = Number(incident.latitude);
                const lon = Number(incident.longitude);
                link.className = "secondary-btn map-focus-btn";
                link.textContent = "Open map";
                link.href = "https://www.openstreetmap.org/?mlat=" + lat +
                    "&mlon=" + lon + "#map=16/" + lat + "/" + lon;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                article.appendChild(link);
            }
            return article;
        }

        async function refreshIncidents() {
            try {
                const response = await fetch(api.active, { headers: { "Accept": "application/json" } });
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.message || "Unable to refresh emergency locations.");
                }
                const incidents = result.incidents || [];
                count.textContent = String(incidents.length);
                list.replaceChildren();
                mapElement.hidden = incidents.length === 0;
                if (incidents.length && !incidentMap) {
                    incidentMap = createMap(
                        mapElement,
                        Number(incidents[0].latitude),
                        Number(incidents[0].longitude),
                        12
                    );
                }

                if (incidents.length === 0) {
                    const empty = document.createElement("p");
                    empty.className = "empty-incidents";
                    empty.textContent = "No active incidents are sharing a location right now.";
                    list.appendChild(empty);
                } else {
                    incidents.forEach(function (incident) {
                        list.appendChild(createIncidentRow(incident));
                    });
                }

                const liveIds = new Set();
                incidents.forEach(function (incident) {
                    const id = String(incident.id);
                    const latitude = Number(incident.latitude);
                    const longitude = Number(incident.longitude);
                    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return;
                    liveIds.add(id);

                    if (incidentMap) {
                        const point = [latitude, longitude];
                        const marker = markers.get(id);
                        if (marker) {
                            marker.setLatLng(point);
                            marker.setPopupContent(popupContent(incident));
                        } else {
                            const newMarker = L.circleMarker(point, {
                                radius: 8,
                                color: "#ffffff",
                                weight: 2,
                                fillColor: "#d94a4a",
                                fillOpacity: 1
                            }).addTo(incidentMap);
                            newMarker.bindPopup(popupContent(incident));
                            markers.set(id, newMarker);
                        }
                    }
                });

                markers.forEach(function (marker, id) {
                    if (!liveIds.has(id)) {
                        marker.remove();
                        markers.delete(id);
                    }
                });

                if (incidentMap && incidents.length && markers.size) {
                    incidentMap.fitBounds(L.featureGroup(Array.from(markers.values())).getBounds(), {
                        padding: [30, 30],
                        maxZoom: 14
                    });
                    window.setTimeout(function () {
                        incidentMap.invalidateSize();
                    }, 0);
                }
            } catch (error) {
                console.error("Unable to refresh emergency locations:", error);
            }
        }

        refreshIncidents();
        window.setInterval(refreshIncidents, 15000);
    }

    initializeOwnerTracking();
    const responderDirectory = initializeResponderDirectory();
    initializeIncidentFeed("responderMap", "responderIncidentList", "responderIncidentCount");
    initializeIncidentFeed("dashboardEmergencyMap", "dashboardIncidentList", "dashboardIncidentCount");
    if (responderDirectory) {
        window.setInterval(responderDirectory.render, 15000);
    }
})();
