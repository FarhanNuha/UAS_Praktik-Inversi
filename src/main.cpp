#include "MainWindow.h"
#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    app.setApplicationName("IHG");
    app.setApplicationVersion("1.0");
    app.setOrganizationName("STMKG");
    
    MainWindow mainWindow;
    mainWindow.show();
    
    return app.exec();
}
