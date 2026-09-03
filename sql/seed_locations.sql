INSERT INTO locations (name, latitude, longitude, notes) VALUES
    ('Phoenix, AZ, USA', 33.44840, -112.07400, 'desert/high-solar, easy case'),
    ('Melbourne, AU', -37.81360, 144.96310, 'temperate/variable (Cfb), medium case'),
    ('Jakarta, ID', -6.20880, 106.84560, 'tropical/equatorial/monsoonal, harder case')
ON CONFLICT (name) DO NOTHING;
