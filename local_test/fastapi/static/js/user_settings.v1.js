document.addEventListener("DOMContentLoaded", () => {
    const userId = document.body.dataset.userId; // 從 HTML 中獲取 user_id
    
    const englishLevelButtons = document.querySelectorAll('[data-type="english_level"]');
    const notificationEnabledButtons = document.querySelectorAll('[data-type="notification_enabled"]');
    const weekButtons = document.querySelectorAll(".week-button");
    const hourSelect = document.getElementById("hour");
    const minuteSelect = document.getElementById("minute");

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

    function updateNotificationTime() {
        fetch("/api/user_settings/update/user_notification", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_id: userId,
                type: "notification_time",
                value: `${hourSelect.value.padStart(2, '0')}:${minuteSelect.value.padStart(2, '0')}`
            })
        }).then(response => response.json())
          .then(data => console.log("Success:", data))
          .catch(error => console.error("Error:", error));
    }


    englishLevelButtons.forEach(button => {
        button.addEventListener("click", () => {
            // 更新按鈕選中狀態
            englishLevelButtons.forEach(btn => btn.classList.remove("selected"));
            button.classList.toggle("selected");

            // 收集資訊並發送到後端
            const settingType = button.dataset.type; // 設定類型
            // 將 按鈕值如:"A1 - A2" 轉換為實際儲存格式如: "$A1-A2"
            let settingValue = button.textContent.trim(); // 按鈕值，先去除多餘的空格
            settingValue = `$${settingValue.replace(/\s+/g, '').replace('-', '-')}`; 

            fetch("/api/user_settings/update/user_settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    user_id: userId,
                    type: settingType,
                    value: settingValue
                })
            }).then(response => response.json())
                .then(data => console.log("Success:", data))
                .catch(error => console.error("Error:", error));
        });
    });

    notificationEnabledButtons.forEach(button => {
        button.addEventListener("click", () => {
            // 更新按鈕選中狀態
            notificationEnabledButtons.forEach(btn => btn.classList.remove("selected"));
            button.classList.toggle("selected");
            
            let isEnabled = button.textContent.trim() === "是";

            // 更新通知相關的元素狀態
            toggleNotificationSettings(isEnabled);

            // 發送後端更新
            fetch("/api/user_settings/update/user_notification", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    user_id: userId,
                    type: "notification_enabled",
                    value: isEnabled ? "True" : "False"
                })
            }).then(response => response.json())
              .then(data => console.log("Success:", data))
              .catch(error => console.error("Error:", error));
        });
    });

    weekButtons.forEach(button => {
        button.addEventListener("click", () => {
            // 更新按鈕選中狀態
            button.classList.toggle("selected");
            
            // 收集資訊並發送到後端
            const settingType = button.dataset.type; // 設定類型
            // 收集所有選中的星期
            let settingValue = "";
            weekButtons.forEach(btn => {
                // 根據按鈕選中狀態生成二進位字符串
                settingValue += btn.classList.contains("selected") ? "1" : "0";
            });

            fetch("/api/user_settings/update/user_notification", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    user_id: userId,
                    type: settingType,
                    value: settingValue
                })
            }).then(response => response.json())
                .then(data => console.log("Success:", data))
                .catch(error => console.error("Error:", error));
        });
    });

    hourSelect.addEventListener("change", updateNotificationTime);
    minuteSelect.addEventListener("change", updateNotificationTime);

    toggleNotificationSettings(notificationEnabledButtons[0].classList.contains("selected"));
});
