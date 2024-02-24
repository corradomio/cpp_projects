#include <iostream>
#include <windows.h>
#include <winuser.h>
#include <wingdi.h>
#include <shellscalingapi.h>

BOOL monitorenumproc(
    HMONITOR hMonitor,
    HDC hDC,
    LPRECT pRect,
    LPARAM pParam
)
{
    printf("Monitor %d\n", hMonitor);
    return TRUE;
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    printf("%d\n", ::GetSystemMetrics(SM_CMONITORS));

    RECT rect;

    SetProcessDPIAware(); //true
    HDC screen = GetDC(NULL);
    double hPixelsPerInch = GetDeviceCaps(screen,LOGPIXELSX);
    double vPixelsPerInch = GetDeviceCaps(screen,LOGPIXELSY);
    ReleaseDC(NULL, screen);
    printf("%d, %d\n", (int)hPixelsPerInch, (int)vPixelsPerInch);

    ::EnumDisplayMonitors(screen, &rect, monitorenumproc, NULL);
    return 0;
}
