module com.example.leetcode {
    requires javafx.controls;
    requires javafx.fxml;


    opens com.example.leetcode to javafx.fxml;
    exports com.example.leetcode;
}