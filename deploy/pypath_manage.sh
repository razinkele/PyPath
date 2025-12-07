#!/bin/bash
#
# PyPath Service Management Script for Shiny Server
# Convenient commands for managing the deployed application on laguna.ku.lt
#

APP_NAME="pypath"
SHINY_SERVER_DIR="/srv/shiny-server"
APP_DIR="${SHINY_SERVER_DIR}/${APP_NAME}"
LOG_DIR="/var/log/shiny-server"

case "$1" in
    start)
        echo "Starting Shiny Server..."
        sudo systemctl start shiny-server
        sleep 2
        sudo systemctl status shiny-server --no-pager
        ;;
    
    stop)
        echo "Stopping Shiny Server..."
        sudo systemctl stop shiny-server
        ;;
    
    restart)
        echo "Restarting Shiny Server..."
        sudo systemctl restart shiny-server
        sleep 2
        sudo systemctl status shiny-server --no-pager
        ;;
    
    status)
        echo "=== Shiny Server Status ==="
        sudo systemctl status shiny-server --no-pager
        echo ""
        echo "=== PyPath App Directory ==="
        ls -la ${APP_DIR}/ 2>/dev/null || echo "App not installed"
        ;;
    
    logs)
        echo "Shiny Server logs (Ctrl+C to exit):"
        sudo tail -f ${LOG_DIR}/shiny-server.log
        ;;
    
    logs-app)
        echo "PyPath application logs:"
        if [[ -d "${LOG_DIR}/${APP_NAME}" ]]; then
            sudo tail -f ${LOG_DIR}/${APP_NAME}/*.log
        else
            echo "App-specific log directory not found"
            echo "Showing main Shiny Server log:"
            sudo tail -f ${LOG_DIR}/shiny-server.log
        fi
        ;;
    
    test)
        echo "Testing application health..."
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3838/pypath/ 2>/dev/null)
        if [[ "$HTTP_CODE" == "200" ]]; then
            echo "✓ Application is responding (HTTP $HTTP_CODE)"
        else
            echo "✗ Application may not be responding (HTTP $HTTP_CODE)"
            echo "Try accessing: http://laguna.ku.lt/pypath"
        fi
        ;;
    
    shell)
        echo "Opening Python shell with app environment..."
        source ${APP_DIR}/venv/bin/activate
        export PYTHONPATH="${APP_DIR}/src"
        cd ${APP_DIR}
        python
        ;;
    
    update)
        if [[ -f "/tmp/pypath_deploy.tar.gz" ]]; then
            echo "Updating from /tmp/pypath_deploy.tar.gz..."
            cd /tmp
            rm -rf pypath_deploy
            tar -xzf pypath_deploy.tar.gz
            cd pypath_deploy
            sudo ./deploy.sh --update
        else
            echo "No update package found at /tmp/pypath_deploy.tar.gz"
            echo "Upload new package first: scp pypath_deploy.tar.gz user@laguna.ku.lt:/tmp/"
        fi
        ;;
    
    backup)
        BACKUP_FILE="/tmp/pypath_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
        echo "Creating backup: ${BACKUP_FILE}"
        sudo tar -czf ${BACKUP_FILE} -C ${SHINY_SERVER_DIR} ${APP_NAME}
        echo "Backup created: ${BACKUP_FILE}"
        ;;
    
    config)
        echo "=== Shiny Server Configuration ==="
        cat /etc/shiny-server/shiny-server.conf
        ;;
    
    edit-config)
        sudo nano /etc/shiny-server/shiny-server.conf
        ;;
    
    *)
        echo "PyPath Service Manager (Shiny Server)"
        echo ""
        echo "Usage: $0 {command}"
        echo ""
        echo "Commands:"
        echo "  start       - Start Shiny Server"
        echo "  stop        - Stop Shiny Server"
        echo "  restart     - Restart Shiny Server"
        echo "  status      - Show server and app status"
        echo "  logs        - Follow Shiny Server logs"
        echo "  logs-app    - Follow PyPath application logs"
        echo "  test        - Test if application is responding"
        echo "  shell       - Open Python shell with app environment"
        echo "  update      - Update from /tmp/pypath_deploy.tar.gz"
        echo "  backup      - Create backup of current installation"
        echo "  config      - Show Shiny Server configuration"
        echo "  edit-config - Edit Shiny Server configuration"
        ;;
esac
