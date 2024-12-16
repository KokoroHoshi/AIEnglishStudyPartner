document.addEventListener("DOMContentLoaded", () => {
    const userId = document.body.dataset.userId; // 從 HTML 中獲取 user_id
    const buttonGroups = document.querySelectorAll(".button-group");
    const weekButtons = document.querySelectorAll(".week-button");
    const hourSelect = document.getElementById("hour");
    const minuteSelect = document.getElementById("minute");
    const notificationEnabledButtons = document.querySelectorAll('[data-type="notification_enabled"]');
    const initialNotificationEnabled = notificationEnabledButtons[0].classList.contains("selected");

    toggleNotificationSettings(initialNotificationEnabled);

    notificationEnabledButtons.forEach(button => {
        button.addEventListener("click", () => {
            const isEnabled = button.textContent.trim() === "是";
            
            // 更新通知相關的元素狀態
            toggleNotificationSettings(isEnabled);

            // 發送後端更新
            fetch("/api/settings/update", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    user_id: userId,
                    setting_type: "notification_enabled",
                    setting_value: isEnabled
                })
            }).then(response => response.json())
              .then(data => console.log("Success:", data))
              .catch(error => console.error("Error:", error));
        });
    });

    function toggleNotificationSettings(isEnabled) {
        // 處理通知星期的按鈕
        weekButtons.forEach(button => {
            if (isEnabled) {
                button.classList.remove("disabled");
                // button.style.pointerEvents = "auto"; // 恢復按鈕點擊
                // button.style.opacity = "1";
            } else {
                button.classList.add("disabled");
                // button.classList.remove("selected"); // 取消選中狀態
            }
        });

        // 處理通知時間選擇框
        hourSelect.disabled = !isEnabled;
        minuteSelect.disabled = !isEnabled;
    }

    buttonGroups.forEach(group => {
        const buttons = group.querySelectorAll(".button");

        buttons.forEach(button => {
            button.addEventListener("click", () => {
                // 收集資訊並發送到後端
                const settingType = button.dataset.type; // 設定類型
                let settingValue = button.textContent; // 按鈕值 (通知星期的按鈕會再次更新)

                if (group.classList.contains("single-select")) {
                    // 讓同一組的其他按鈕失去選中狀態
                    buttons.forEach(btn => btn.classList.remove("selected"));
                }

                if (button.classList.contains("week-button")) {
                    // 收集所有選中的星期
                    let selectedDaysBinary = "";
                    weekButtons.forEach(btn => {
                        // 根據按鈕選中狀態生成二進位字符串
                        selectedDaysBinary += btn.classList.contains("selected") ? "1" : "0";
                    });

                    settingValue = selectedDaysBinary;
                }

                // 更新按鈕選中狀態
                button.classList.toggle("selected");

                fetch("/api/settings/update", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_id: userId,
                        setting_type: settingType,
                        setting_value: settingValue
                    })
                }).then(response => response.json())
                  .then(data => console.log("Success:", data))
                  .catch(error => console.error("Error:", error));
            });
        });
    });

    // 當時間或分鐘改變時，發送設定
    if (hourSelect && minuteSelect) {
        hourSelect.addEventListener("change", updateNotificationTime);
        minuteSelect.addEventListener("change", updateNotificationTime);
    }

    function updateNotificationTime() {
        const selectedHour = hourSelect.value;
        const selectedMinute = minuteSelect.value;

        fetch("/api/settings/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_id: userId,
                setting_type: "notification_time",
                setting_value: `${selectedHour}:${selectedMinute}`
            })
        }).then(response => response.json())
          .then(data => console.log("Success:", data))
          .catch(error => console.error("Error:", error));
    }
});
